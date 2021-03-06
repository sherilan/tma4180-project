
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.spatial
import seaborn as sns

sns.set_style('darkgrid')


#%%
class Problem:

    def __init__(self, A, v, objective, title=None):
        assert len(A.shape) == 2, '`A` argument must be a matrix'
        assert len(v.shape) == 1, '`v` argument must be a vector'
        assert A.shape[0] == v.shape[0], '`A` and `v` first axis must match'
        assert A.shape[1] == 2, '`A` must be an mx2 matrix'
        self.objective = objective
        self.A = A
        self.v = v
        self.f = objective
        self.m = A.shape[0]
        self.conv = scipy.spatial.ConvexHull(self.A)
        self.title = title or f'{self.m} locations'

    def __call__(self, x):
        return self.f(self.A, self.v, np.asarray(x))

    def get_centroid(self, weighted=True):
        if weighted:
            w = self.v
        else:
            w = np.ones(self.m)
        return np.average(self.A, weights=w, axis=0)

    @classmethod
    def premade(cls, name, **kwargs):
        options = {
            'equilateral': (
                [(-1, -(3**0.5)/2), (+1, -(3**0.5)/2), (0, +3**0.5/2)],
                [1] * 3,
            ),
            'five': (
                [(-1, -1), (+1, -1), (-1, +1), (+1, +1), (0, 0)],
                [1] * 5,
            ),
            'box': (
                [(-1, -1), (+1, -1), (-1, +1), (+1, +1)],
                [1] * 4,
            ),
            'rectangle': (
                [(-0.1, -1), (+0.1, -1), (-0.1, +1), (+0.1, +1)],
                [1] * 4,
            ),
            'uneven_rectangle': (
                [(-0.1, -1), (+0.1, -1), (-0.1, +1), (+0.1, +1)],
                [2, 2, 1, 1],
            ),
            'outlier': (
                [(-0.5, -0.5), (-0.5, +0.5), (+0.5, -0.5), (+0.5, +0.5), (3, 0)],
                [1, 1, 1, 1, 4],
            ),
        }
        if not name in options:
            raise ValueError(
                f'Bad problem name "{name}". Choose among: {list(options)}'
            )
        A, v = map(np.array, options[name])
        return cls(A, v, **kwargs)

    @classmethod
    def generate(cls,
        m,
        a_min=-1,
        a_max=1,
        v_min=0.1,
        v_max=1.0,
        a_min_space=None,
        seed=None,
        **kwargs
    ):
        # Create PRNG if seed provided
        random = np.random if seed is None else np.random.RandomState(seed)
        # Sample A and V
        A = random.uniform(a_min, a_max, size=(m, 2))
        v = random.uniform(v_min, v_max, size=m)
        return cls(A, v, **kwargs)

    def __repr__(self):
        return f'<Problem obj={self.objective} m={self.m}>'


    def _repr_html_(self):
        df = pd.DataFrame()
        df['a_0'] = self.A[:,0]
        df['a_1'] = self.A[:,1]
        df['v'] = self.v
        return df.T.to_html()


    def plot_2d(
        self,
        iterates=None,
        ax=None,
        x0_min=None,
        x0_max=None,
        x1_min=None,
        x1_max=None,
        res=100,
        cmap='Blues_r',
        verbose=False,
        legend=None,
        figsize=None,
    ):
        if ax is None:
            if not figsize is None:
                plt.figure(figsize=figsize)
            ax = plt.gca()

        legend = self.m <= 20 if legend is None else legend

        # Determine viewpoirt
        x0_min = min(self.A[:,0].min(), -1) * 1.2 if x0_min is None else x0_min
        x0_max = max(self.A[:,0].max(), +1) * 1.2 if x0_max is None else x0_max
        x1_min = min(self.A[:,1].min(), -1) * 1.2 if x1_min is None else x1_min
        x1_max = max(self.A[:,1].max(), +1) * 1.2 if x1_max is None else x1_max
        x0_delta = x0_max - x0_min
        x1_delta = x1_max - x1_min

        res0 = int(res * max(1, x0_delta / x1_delta))
        res1 = int(res * max(1, x1_delta / x0_delta))

        # Create viewport transform (since heatmap use raster pixel coordinates)
        def map_x0(x0):
            return res0 * (x0 - x0_min) / x0_delta

        def map_x1(x1):
            return res1 * (x1 - x1_min) / x1_delta

        # Rasterize
        x0 = np.linspace(x0_min, x0_max, res0)
        x1 = np.linspace(x1_min, x1_max, res1)
        X0, X1 = np.meshgrid(x0, x1)
        X = np.stack([X0, X1], axis=-1)
        L = self(X)

        # Make heatmap with contour
        ax = ax or plt.gca()
        ax.imshow(L, cmap=cmap, alpha=0.7)
        ax.grid(color='white', linestyle='-', linewidth=1, alpha=0.1)
        vmin, vmax = L.min(), L.max() * 2
        contours = ax.contour(
            L, cmap=cmap, vmin=vmin, vmax=vmax, linestyles='--', linewidths=1.0
        )
        ax.clabel(contours, inline=1, fontsize=10)

        # x0 ticks
        x0_tick_labels = (
            np.arange(x0_min, x0_max + 1e-6, (x0_max - x0_min)/12)
              .round(int(np.ceil(np.log10(12/2))))
        )
        x0_ticks = map_x0(x0_tick_labels)
        ax.set_xticks(x0_ticks)
        ax.set_xticklabels(x0_tick_labels)

        # x1 ticks
        x1_tick_labels = (
            np.arange(x1_min, x1_max + 1e-6, (x1_max - x1_min)/12)
              .round(int(np.ceil(np.log10(12/2))))
        )
        x1_ticks = map_x1(x1_tick_labels)
        ax.set_yticks(x1_ticks)
        ax.set_yticklabels(x1_tick_labels)

        # Plot As
        colors = sns.color_palette(n_colors=self.m)
        if self.m > 20:
            colors = [colors[0]] * self.m

        for i, ((a0, a1), v, c) in enumerate(zip(self.A, self.v, colors)):
            if legend:
                if verbose:
                    label = f'$a^{{{i}}}$ ($x_0={a0:.2f}$, $x_1={a1:.2f}$, $v={v:.2f}$)'
                else:
                    label = f'$a^{{{i}}}$ ($v^{{{i}}}={v:.2f}$)'
            else:
                label = None
            ax.scatter(
                map_x0(a0), map_x1(a1),
                s=25 + 75 * v,
                color=c,
                edgecolors='0',
                label=label,
                zorder=2,
            )

        # Plot convex hull of As
        ch_indices = list(self.conv.vertices) + [self.conv.vertices[0]]
        ch_path0 = self.A[:,0][ch_indices]
        ch_path1 = self.A[:,1][ch_indices]
        ax.plot(map_x0(ch_path0), map_x1(ch_path1), '--', color='w', alpha=0.5, zorder=1)

        # Plot iterates if provided
        if not iterates is None and len(iterates):
            if isinstance(iterates, pd.DataFrame):
                iterates = iterates[['x0', 'x1']]
            iterates = np.asarray(iterates)
            it0 = map_x0(iterates[:,0])
            it1 = map_x0(iterates[:, 1])
            # ax.scatter(it0[:1], it1[:1], marker='x', color='red', label='$x_0$')
            ax.plot(it0, it1, '+-', color='red', alpha=0.6, zorder=3) # label='$x_i$'
            ax.scatter(
                it0[-1:], it1[-1:],
                s=200,
                marker='*',
                color='red',
                label='$x^t$',
                zorder=4,
                edgecolors='0'
            )

        # Add legend and labels
        # ax.legend(loc='upper center', bbox_to_anchor=(0.5,1.15), ncol=4, fancybox=True)
        if legend:
            ax.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))
        else:
            ax.legend().remove()
        ax.set_xlabel('$x_0$')
        ax.set_ylabel('$x_1$')

        return ax

    def plot_3d(
        self,
        iterates=None,
        ax=None,
        x0_min=None,
        x0_max=None,
        x1_min=None,
        x1_max=None,
        res=100,
        cmap='Blues_r',
        legend=None,
        figsize=None,
    ):
        legend = self.m <= 20 if legend is None else legend

        # Determine viewpoirt
        x0_min = min(self.A[:,0].min(), -1) * 1.2 if x0_min is None else x0_min
        x0_max = max(self.A[:,0].max(), +1) * 1.2 if x0_max is None else x0_max
        x1_min = min(self.A[:,1].min(), -1) * 1.2 if x1_min is None else x1_min
        x1_max = max(self.A[:,1].max(), +1) * 1.2 if x1_max is None else x1_max

        # Rasterize
        x0 = np.linspace(x0_min, x0_max, res)
        x1 = np.linspace(x1_min, x1_max, res)
        X0, X1 = np.meshgrid(x0, x1)
        X = np.stack([X0, X1], axis=-1)
        L = self(X)

        # Make surface plot as backdrop
        if ax is None:
            if not figsize is None:
                plt.figure(figsize=figsize)
            ax = plt.gca(projection='3d')
        ax.plot_surface(
            X0, X1, L,
            rstride=1,
            cstride=1,
            cmap=cmap,
            edgecolor='none',
            alpha=0.6,
        )

        # Plot As
        colors = sns.color_palette(n_colors=self.m)
        ls = self(self.A)
        for i, ((a0, a1), l, v, c) in enumerate(zip(self.A, ls, self.v, colors)):
            ax.scatter(
                a0, a1, l,
                s=25 + 75 * v,
                color=c,
                edgecolors='0',
                label=f'$a_{{{i}}}$ (${v:.2f}$)'
            )

        # Plot convex hull of As
        ch_indices = list(self.conv.vertices) + [self.conv.vertices[0]]
        ch_path0 = self.A[:,0][ch_indices]
        ch_path1 = self.A[:,1][ch_indices]
        ch_l = ls[ch_indices]
        ax.plot(ch_path0, ch_path1, ch_l, '--', color='#343434', alpha=0.3)

        # Plot iterates if provided
        if not iterates is None and len(iterates):
            if isinstance(iterates, pd.DataFrame):
                iterates = iterates[['x0', 'x1']]
            iterates = np.asarray(iterates)
            it0 = iterates[:,0]
            it1 = iterates[:, 1]
            itl = self(iterates)
            # ax.scatter(it0[:1], it1[:1], marker='x', color='red', label='$x_0$')
            ax.plot(it0, it1, itl, '+-', color='red', alpha=0.6) # label='$x_i$'
            # ax.scatter(it0[-1:], it1[-1:], marker='*', color='red', label='$x_n$')

        # Add legend and labels
        if legend:
            ax.legend(loc='upper center', bbox_to_anchor=(0.5,1.15), ncol=4, fancybox=True)
        ax.set_xlabel('$x_0$')
        ax.set_ylabel('$x_1$')
        ax.set_zlabel('$f(x_0, x_1)$')
