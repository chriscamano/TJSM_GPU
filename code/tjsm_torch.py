
import matplotlib.pyplot as plt 
import torch
import concurrent.futures
import numpy as np 
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable


TOL=1e-16 #Change for looser masking
class TracialSpectralDensity:
    def __init__(self, C, resolution=256, zoom_scale=1.0, p_scale=3.0, chunk_size=10000,
                 cmap_name="viridis", log_scale=False):
        """
        Initialize the object with matrix C and plotting parameters.

        Parameters:
            C : torch.Tensor
                The input matrix (can be complex).
            resolution : int
                Number of grid points along each axis.
            zoom_scale : float
                Zoom factor used to set the domain limits.
            p_scale : float
                Additional scaling for the domain limits.
            chunk_size : int
                Number of grid points processed per chunk.
            cmap_name : str
                Colormap for the plot.
            log_scale : bool
                If True, use logarithmic scaling for the density.
        """
        self.C = C
        self.resolution = resolution
        self.zoom_scale = zoom_scale
        self.p_scale = p_scale
        self.chunk_size = chunk_size
        self.cmap_name = cmap_name
        self.log_scale = log_scale

        # Cache computed quantities.
        self._domain_limit = None
        self._a_vals = None
        self._b_vals = None
        self._grid_a = None
        self._grid_b = None
        self._A = None
        self._B = None
        self._density = None
        self._singular_points = None

        self.device = self.C.device
        self.d = self.C.shape[0]
        self._compute_domain()
        self._compute_matrices()
        self._compute_meshgrid()
    
    def _compute_domain(self):
        """Compute the fixed domain limit."""
        self._domain_limit = self.zoom_scale * self.d * self.p_scale
    
    def _compute_matrices(self):
        """
        Computes the full matrices A and B from C:
            A = (C + C^*)/2,   B = -i*(C - C^*)/2.
        We store the full matrices, not just their real parts.
        """
        self._A = (self.C + self.C.conj().T) / 2
        self._B = -1j * (self.C - self.C.conj().T) / 2
    
    def _compute_meshgrid(self):
        """Construct a static mesh grid over \([-L, L]^2\)."""
        L = self._domain_limit
        self._a_vals = torch.linspace(-L, L, steps=self.resolution, device=self.device)
        self._b_vals = torch.linspace(-L, L, steps=self.resolution, device=self.device)
        self._grid_a, self._grid_b = torch.meshgrid(self._a_vals, self._b_vals, indexing='ij')
    
    def _compute_density_chunked(self, A_full, B_full):
        """
        Computes the spectral density on the fixed mesh grid using a chunked approach.
        """
        N = self._grid_a.shape[0]
        num_points = N * N
        d = A_full.shape[0]
        identity = torch.eye(d, dtype=A_full.dtype, device=A_full.device)

        flat_a = self._grid_a.reshape(-1)
        flat_b = self._grid_b.reshape(-1)
        denominator = flat_a**2 + flat_b**2

        def process_chunk(start, end):
            a_chunk = flat_a[start:end].view(-1, 1, 1)
            b_chunk = flat_b[start:end].view(-1, 1, 1)
            denom_chunk = denominator[start:end].view(-1, 1, 1)
            
            A_exp = A_full.unsqueeze(0).expand(a_chunk.size(0), d, d)
            B_exp = B_full.unsqueeze(0).expand(a_chunk.size(0), d, d)
            id_exp = identity.unsqueeze(0).expand(a_chunk.size(0), d, d)

            first_term = id_exp - (a_chunk / denom_chunk) * A_exp - (b_chunk / denom_chunk) * B_exp
            second_term = b_chunk * A_exp - a_chunk * B_exp

            M = torch.linalg.solve(second_term.transpose(-2, -1),
                                     first_term.transpose(-2, -1)).transpose(-2, -1)
            
            use_eigh = not torch.is_complex(self.C)
            if use_eigh:
                eigvals = torch.linalg.eigh(M, eigenvectors=False).to(torch.complex64)
            else:
                eigvals = torch.linalg.eigvals(M)
            imag_part = torch.abs(eigvals.imag)
            mask = imag_part > TOL
            density_chunk = torch.sum(torch.where(mask, imag_part, torch.zeros_like(imag_part)), dim=-1)
            return density_chunk

        chunks = [(i, min(i + self.chunk_size, num_points)) for i in range(0, num_points, self.chunk_size)]
        density_chunks = [None] * len(chunks)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(process_chunk, start, end) for start, end in chunks]
            for idx, future in enumerate(futures):
                density_chunks[idx] = future.result()

        density_total = torch.cat(density_chunks, dim=0)
        return density_total.view(self._grid_a.shape[0], self._grid_a.shape[1])
    
    def compute_density(self):
        """Compute and cache the spectral density on the fixed mesh grid."""
        if self._density is None:
            self._density = self._compute_density_chunked(self._A, self._B)
        return self._density, self._a_vals.cpu().numpy(), self._b_vals.cpu().numpy()

    def compute_singular_points(self):
        """
        Computes singular points in high precision using the full matrices A and B.
        For each eigenvector \( v \) of \( M = A^{-1} B \) (with \( A \) and \( B \) in high precision),
        the singular point is computed via the Rayleigh quotients:
            \( a_0 = \Re(v^* A v), \quad b_0 = \Re(v^* B v) \).
        All eigenvectors are used.
        """
        if self._singular_points is None:
            # Convert the full matrices to high precision as complex128.
            A_hp = self._A.to(torch.complex128)
            B_hp = self._B.to(torch.complex128)
            try:
                A_inv_hp = torch.linalg.inv(A_hp)
            except RuntimeError:
                print("Matrix A is not invertible. Singular point computation aborted.")
                self._singular_points = []
                return self._singular_points
            
            M_hp = A_inv_hp @ B_hp
            _, eigenvectors = torch.linalg.eig(M_hp)
            singular_points = []
            for idx in range(eigenvectors.shape[1]):
                vec = eigenvectors[:, idx]
                vec = vec / torch.linalg.norm(vec)
                # Compute the Rayleigh quotients using the full high-precision matrices.
                a_val = torch.real(torch.dot(vec.conj(), A_hp @ vec))
                b_val = torch.real(torch.dot(vec.conj(), B_hp @ vec))
                singular_points.append((a_val.item(), b_val.item()))
            self._singular_points = singular_points
        return self._singular_points

    def visualize(self, return_fig=False):
        """
        Produces a plot in the fixed \(\mathbb{R}^2\) domain.
        The density is plotted on the mesh grid, and singular points with associated lines are overlaid.
        
        Parameters:
            return_fig : bool
                If True, returns the matplotlib figure object instead of calling plt.show().
        """
        density, a_vals, b_vals = self.compute_density()
        density_np = density.cpu().numpy()
        # Transpose density so that density_plot[y,x] corresponds to (a, b).
        density_plot = density_np.T

        L = self._domain_limit
        a_min, a_max = -L, L
        b_min, b_max = -L, L

        singular_points = self.compute_singular_points()

        fig = plt.figure(figsize=(10, 10))
        ax = fig.gca()
        
        # Set normalization if log scale is enabled.
        if self.log_scale:
            pos_density = density_plot[density_plot > 0]
            vmin = pos_density.min() if pos_density.size > 0 else 1e-10  # Avoid zero division
            vmax = density_plot.max()
            norm = LogNorm(vmin=vmin, vmax=vmax)
        else:
            norm = None

        # Plot the density image and capture the returned image object.
        im = ax.imshow(density_plot,
                    extent=[a_min, a_max, b_min, b_max],
                    origin='lower',
                    cmap=self.cmap_name,
                    norm=norm,
                    aspect='equal')
        
        # Increase tick label fontsize on the axes.
        ax.tick_params(axis='both', labelsize=14)

        # Use an axes divider to attach a colorbar that matches the axes height.
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cb = plt.colorbar(im, cax=cax)
        cb.set_label(r"Spectral Density $\rho(a,b)$", fontsize=16)
        cb.ax.tick_params(labelsize=16)

        # Update the title to include the matrix dimension.
        ax.set_title(f"Tracial Joint Spectral Density, Matrix dimension n={self.d}", fontsize=16)
        ax.set_xlabel("$a$", fontsize=14)
        ax.set_ylabel("$b$", fontsize=14)

        # Plot singular lines and points in 'cyan' for better visibility.
        for (a0, b0) in singular_points:
            if not (a_min <= a0 <= a_max and b_min <= b0 <= b_max):
                continue
            t_max_a = a_max / abs(a0) if a0 != 0 else np.inf
            t_max_b = b_max / abs(b0) if b0 != 0 else np.inf
            t_max = min(t_max_a, t_max_b)
            t_vals = np.linspace(-t_max, t_max, 1000)
            line_a = t_vals * a0
            line_b = t_vals * b0
            line_handle, = ax.plot(line_a, line_b, color='cyan', alpha=0.3, linewidth=1.5)
            line_handle.set_dashes([5, 5])
            ax.scatter(a0, b0, color='cyan', s=60, alpha=0.4, edgecolor='black', linewidth=1.5, zorder=5)

        # Create custom legend entries.
        legend_line = plt.Line2D([], [], color='cyan', alpha=1, linewidth=1.5, linestyle=(0, (5, 5)))
        legend_point = plt.Line2D([], [], color='cyan', marker='o', markersize=8,
                                    markeredgecolor='black', alpha=1, markeredgewidth=1.5, linestyle='None')
        ax.legend([legend_line, legend_point],
                ["Singular Lines", "Singular Points"],
                loc='upper center',
                bbox_to_anchor=(0.5, 1.12),
                ncol=2,
                fancybox=True,
                shadow=True,
                fontsize=16)

        plt.tight_layout()
        
        if return_fig:
            return fig
        else:
            plt.show()

#Example.
if __name__ == '__main__':
    device = torch.device('cpu')
    matrix_dim = 4
    real_part = torch.randn((matrix_dim, matrix_dim), device=device)
    imag_part = torch.randn((matrix_dim, matrix_dim), device=device)
    C = real_part + 1j * imag_part

    tsd = TracialSpectralDensity(C,
                                 resolution=1000,
                                 zoom_scale=1.05,
                                 p_scale=1.0,
                                 chunk_size=10000,
                                 cmap_name="magma",
                                 log_scale=True)
    tsd.visualize()