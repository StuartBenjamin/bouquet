"""bouquet main's defaults where PSFC_dev changes them (40fdc5d).  The tests
that check main's behaviour pass these explicitly, so they test the same code
on both branches."""
#: ReconstructionSource
RECON = dict(ni_source="standard")
#: ImasSource
IMAS = dict(ida_timing="window", zeff_fast_ions=False, psi_pad=1e-3)
#: GenerationConfig
GEN = dict(swb_edge_taper_psi0=None, compress_file_bytes=False,
           ifile_profile_derivs=False, ifile_psi_bounds=False)


def from_imas(*args, **kw):
    """``Bouquet.from_imas`` on main's defaults (``kw`` still wins)."""
    import bouquet as bq
    ida_timing = kw.pop("ida_timing", IMAS["ida_timing"])
    psi_pad = kw.pop("psi_pad", IMAS["psi_pad"])
    b = bq.Bouquet.from_imas(*args, **dict(zeff_fast_ions=IMAS["zeff_fast_ions"], **kw))
    b.config.source.ida_timing = ida_timing
    b.config.source.psi_pad = psi_pad
    return b
