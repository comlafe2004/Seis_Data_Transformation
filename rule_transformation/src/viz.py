# viz.py
import os
import numpy as np
import matplotlib.pyplot as plt

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def plot_density(section, out_png, title, dt_ms, tmax_ms, clip=60.0, cmap="seismic", colorbar=True):
    ensure_dir(os.path.dirname(out_png) or ".")
    ns, ntr = section.shape
    extent = [0, ntr-1, tmax_ms, 0]
    v = float(clip) if clip is not None else float(np.nanmax(np.abs(section)))
    fig, ax = plt.subplots(figsize=(10.5, 6), dpi=200)
    im = ax.imshow(section, aspect="auto", cmap=cmap, vmin=-v, vmax=v, extent=extent)
    ax.set_title(title)
    ax.set_xlabel("Trace index")
    ax.set_ylabel("Time (ms)")
    if colorbar:
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Amplitude (a.u.)")
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

def plot_side_by_side(a, b, out_png, titles, dt_ms, tmax_ms, clip=60.0, cmap="seismic"):
    ensure_dir(os.path.dirname(out_png) or ".")
    ns, ntr = a.shape
    extent = [0, ntr-1, tmax_ms, 0]
    v = float(clip) if clip is not None else float(max(np.nanmax(np.abs(a)), np.nanmax(np.abs(b))))
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 6), dpi=200, sharey=True)
    im0 = axes[0].imshow(a, aspect="auto", cmap=cmap, vmin=-v, vmax=v, extent=extent)
    axes[0].set_title(titles[0]); axes[0].set_xlabel("Trace index"); axes[0].set_ylabel("Time (ms)")
    im1 = axes[1].imshow(b, aspect="auto", cmap=cmap, vmin=-v, vmax=v, extent=extent)
    axes[1].set_title(titles[1]); axes[1].set_xlabel("Trace index")
    cbar = fig.colorbar(im1, ax=axes.ravel().tolist(), fraction=0.046, pad=0.04)
    cbar.set_label("Amplitude (a.u.)")
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

def plot_difference(diff, out_png, title, dt_ms, tmax_ms, clip=None, cmap="seismic"):
    ensure_dir(os.path.dirname(out_png) or ".")
    ns, ntr = diff.shape
    extent = [0, ntr-1, tmax_ms, 0]
    v = float(np.nanpercentile(np.abs(diff), 99.5)) if clip is None else float(clip)
    v = max(v, 1e-9)
    fig, ax = plt.subplots(figsize=(10.5, 6), dpi=200)
    im = ax.imshow(diff, aspect="auto", cmap=cmap, vmin=-v, vmax=v, extent=extent)
    ax.set_title(title)
    ax.set_xlabel("Trace index")
    ax.set_ylabel("Time (ms)")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Difference (a.u.)")
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

def plot_wiggle_all_traces(section, out_png, dt_ms, tmax_ms, scale=None, title="Wiggle (all traces)"):
    ensure_dir(os.path.dirname(out_png) or ".")
    ns, ntr = section.shape
    t = np.linspace(0, tmax_ms, ns)
    if scale is None:
        scale = 0.6 / (np.nanpercentile(np.abs(section), 99) + 1e-9)
    fig, ax = plt.subplots(figsize=(14, 7), dpi=200)
    for i in range(ntr):
        x = section[:, i] * scale + i
        ax.plot(x, t, color="black", linewidth=0.3)
        ax.fill_betweenx(t, i, x, where=(x > i), color="black", alpha=0.15, linewidth=0)
    ax.set_ylim(tmax_ms, 0)
    ax.set_xlim(-1, ntr)
    ax.set_title(title)
    ax.set_xlabel("Trace index")
    ax.set_ylabel("Time (ms)")
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

def amplitude_spectrum(section, dt_ms):
    ns, ntr = section.shape
    dt = dt_ms / 1000.0
    nfft = int(2 ** np.ceil(np.log2(ns)))
    spec = np.zeros(nfft//2 + 1, dtype=np.float64)
    for i in range(ntr):
        X = np.fft.rfft(section[:, i].astype(np.float64), n=nfft)
        spec += np.abs(X)
    spec /= ntr
    f = np.fft.rfftfreq(nfft, d=dt)
    return f, spec

def plot_spectra(sections_dict, out_png, dt_ms, fmax=None, title="Amplitude spectra"):
    ensure_dir(os.path.dirname(out_png) or ".")
    fig, ax = plt.subplots(figsize=(10, 5), dpi=200)
    for name, sec in sections_dict.items():
        f, s = amplitude_spectrum(sec, dt_ms)
        if fmax is not None:
            m = f <= fmax
            ax.plot(f[m], s[m], label=name)
        else:
            ax.plot(f, s, label=name)
    ax.set_title(title)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Amplitude (a.u.)")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
