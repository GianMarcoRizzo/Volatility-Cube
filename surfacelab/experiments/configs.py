"""
Thin experiment definitions: which data, which models, which harness mode.

An `Experiment` says how to load its dataset and which registry models to evaluate.
`run.py` consumes these.  CNP entries point at the cached checkpoints under
trained_models/ by default; pass --retrain to train fresh (a CNPTrainConfig is provided).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from surfacelab.data import load_heston, load_grouptech, compute_bspline_prior

REPO = Path(__file__).resolve().parents[2]
HESTON_TRAIN = str(REPO / "data" / "synthetic" / "heston_multiasset_training.npz")
HESTON_OOD = str(REPO / "data" / "synthetic" / "heston_multiasset_ood_test.npz")
MARKET_CSV = str(REPO / "data" / "scripts" / "bulk_download" / "output" / "group_tech_us.csv")
MARKET_HOURLY = str(REPO / "data" / "scripts" / "bulk_download" / "output"
                    / "group_tech_hourly" / "group_tech_us_hourly.csv")
TRAINED = REPO / "trained_models"

# Context-size sweeps (per asset); 500 ≈ whole surface.
DEFAULT_CTX = (2, 4, 8, 10, 20, 40, 80, 100, 200, 300, 500)
ASYM_CTX = (1, 2, 3, 5, 10, 20, 50, 100)


# ── CNP training config (used only when training from scratch) ────────────────
@dataclass
class _ModelCfg:
    # Attentive Neural Process: query points cross-attend to the encoded context (no latent
    # bottleneck, no mean pooling).  See surfacelab/models/cnp/module.py.
    d_asset: int = 16
    d_model: int = 64
    n_heads: int = 4
    n_layers_enc: int = 3      # context self-attention layers
    n_layers_dec: int = 3      # query→context cross-attention layers
    d_hidden: int = 128        # MLP head width
    n_fourier: int = 16        # random Fourier features per (k,T) coord (0 = off)
    fourier_scale: float = 2.0 # std of the random Fourier frequencies
    dropout: float = 0.1


@dataclass
class _TrainCfg:
    n_epochs: int = 20
    batch_size: int = 32
    lr: float = 3e-4
    ctx_min: int = 5
    log_every: int = 10
    # Loss weighting (market data only — needs bid/ask):
    #   "none"   → plain unweighted RMSE; every quote counts equally, model must fit the
    #              illiquid wings too (default — best raw vol-point RMSE, improves with ctx)
    #   "light"  → gentle inverse-spread tilt (clamp 0.5–2×)
    #   "spread" → old heavy vega-adjusted 1/IV-spread² weighting (ATM dominates)
    loss_weighting: str = "none"
    # Optional float override of the preset inverse-spread strength (0 = uniform, 1 = full).
    loss_weight_strength: float | None = None


@dataclass
class CNPTrainConfig:
    device: str = "cuda"
    model: _ModelCfg = field(default_factory=_ModelCfg)
    train: _TrainCfg = field(default_factory=_TrainCfg)


@dataclass
class Experiment:
    name: str
    loader: Callable                      # () -> (Dataset, ood_or_None)
    models: list                          # list of (registry_name, kwargs) tuples
    mode: str = "independent"             # "independent" | "sequential"
    ctx_sizes: tuple = DEFAULT_CTX
    seq_ctx_sizes: tuple = DEFAULT_CTX     # context sweep for sequential mode
    needs_prior: bool = False             # compute B-spline prior on the dataset
    exclude_asset: str | None = None      # leave-one-asset-out: this asset gets no context
    asymmetric_target: str | None = None  # asymmetric-liquidity: peers FULL context, this
                                          # asset gets `ctx_sizes` quotes (the swept x-axis)
    # how much of yesterday a temporal model is seeded with (independent harness only):
    # "full" = yesterday's whole surface (perfect-prior reference); "match" = yesterday
    # sampled at the same n_ctx as today (fair — no information the absolute models lack).
    prior_ctx: str = "match"
    # NEW composable path: given the built+trained fitters keyed by name, return a list of
    # eval.Model bundles run through the unified run_models loop.  When set, it supersedes
    # the legacy mode/exclude/asymmetric dispatch.  `models` is still used to build/train.
    specs: Callable | None = None

    @property
    def out_dir(self) -> str:
        return str(REPO / "results" / "surfacelab" / self.name)


def _heston(n_train=None, n_val=40):
    return lambda: load_heston(HESTON_TRAIN, HESTON_OOD, n_train_days=n_train, n_val_days=n_val)


def _market(n_eval=30):
    return lambda: (load_grouptech(MARKET_CSV, n_eval_days=n_eval), None)


# ── composable Model builders (the new path) ──────────────────────────────────────
def sweep(fitter, ctx_sizes, *, regimes=("unif", "extrap"), prior_ctx="match",
          prior_mode="fit"):
    """One Model per (regime, ctx size) for a built fitter — the today/yesterday splitters
    encode what used to be the harness's regime + prior_ctx + ctx sweep."""
    from surfacelab.eval import Model, Uniform, Extrap, Matched, Full
    out = []
    for nc in ctx_sizes:
        for reg in regimes:
            today = Extrap(nc) if reg == "extrap" else Uniform(nc)
            yest = Matched(nc, reg) if prior_ctx == "match" else Full()
            out.append(Model(fitter=fitter, today=today, yesterday=yest, prior_mode=prior_mode))
    return out


def sweep_asymmetric(fitter, target, ctx_sizes, *, regimes=("unif", "extrap"),
                     prior_ctx="full", prior_mode="fit"):
    """Asymmetric-liquidity Models: peers full, target gets nc quotes, score target only."""
    from surfacelab.eval import Model, Asymmetric, Matched, Full
    out = []
    for nc in ctx_sizes:
        for reg in regimes:
            today = Asymmetric(target, nc, regime=reg)
            yest = Full() if prior_ctx == "full" else Matched(nc, reg)
            out.append(Model(fitter=fitter, today=today, yesterday=yest, prior_mode=prior_mode))
    return out


_CNP_CKPT = {"checkpoint": str(TRAINED / "cnp.pt")}
_CNP_DELTA_CKPT = {"checkpoint": str(TRAINED / "cnp_delta.pt")}

EXPERIMENTS: dict[str, Experiment] = {
    "heston_all_methods": Experiment(
        name="heston_all_methods",
        loader=_heston(n_train=None, n_val=40),
        models=[("prior", {}), ("svi", {}), ("ssvi", {}), ("bspline", {}),
                ("pca", {}),
                ("kalman_pca", {}), ("kalman_ssvi", {}), ("kalman_ssvi_inc", {}),
                ("cnp", _CNP_CKPT), ("cnp_delta", _CNP_DELTA_CKPT)],
        mode="independent",
        needs_prior=True,
    ),
    "heston_all_methods_sequential": Experiment(
        name="heston_all_methods_sequential",
        loader=_heston(n_train=None, n_val=40),
        models=[("prior", {}),                    # persistence: yesterday's surface
                ("bspline_data", {}), ("bspline_temporal", {}),
                ("bspline_temporal_graph", {}),   # uniform M=I edges
                ("bspline_tiered_graph", {}),     # SPY-leads-stock tiered precision
                ("bspline_factored_graph", {}),   # + cross-maturity decay
                ("bspline_learned_graph", {}),    # OLS-learned coupling matrix M
                # …same six + within-asset maturity smoothness (λ_maturity>0) to compare
                ("bspline_data_interp", {}), ("bspline_temporal_interp", {}),
                ("bspline_temporal_graph_interp", {}),
                ("bspline_tiered_graph_interp", {}),
                ("bspline_factored_graph_interp", {}),
                ("bspline_learned_graph_interp", {}),
                ("svi_temporal", {}), ("svi_temporal_graph", {}),
                ("ssvi_data", {}),                # per-day SSVI fit (temporal off)
                ("ssvi_temporal", {}), ("ssvi_temporal_graph", {}),
                ("kalman_pca", {}), ("kalman_ssvi", {}), ("kalman_ssvi_inc", {})],
        mode="sequential",
    ),
    "market_all_methods": Experiment(
        name="market_all_methods",
        loader=_market(n_eval=10),
        models=[("prior", {}),                     # yesterday's full surface = delta-CNP's B-spline prior
                ("bspline_data", {}),             # pure per-day fit → ~0 RMSE as n_ctx grows
                ("bspline_temporal", {}),         # + pull to yesterday (prior help, decays w/ data)
                ("bspline_temporal_graph", {}),   # + uniform cross-asset coupling on increments
                ("bspline_learned_graph", {}),    # + OLS-learned cross-asset coupling matrix M
                ("bspline_market_graph", {}),     # + single-market-factor (SPY) level coupling
                ("bspline_pca_graph", {}),        # + low-rank PCA-mode cross-asset coupling
                ("bspline_temporal_graph_interp", {}),  # + within-asset maturity smoothness
                ("ssvi_data", {}),                # per-day SSVI fit (temporal off)
                ("ssvi_temporal", {}), ("ssvi_temporal_graph", {}),
                ("kalman_pca", {}), ("kalman_ssvi", {}), ("kalman_ssvi_inc", {}),
                # delta-CNP carries its B-spline prior forward via step().
                ("cnp", {"checkpoint": str(TRAINED / "cnp_market.pt")}),
                ("cnp_delta", {"checkpoint": str(TRAINED / "cnp_delta_market.pt")})],
        mode="independent",
        needs_prior=True,
        # composable path: each fitter swept over both regimes × ctx sizes, matched prior.
        specs=lambda F: [m for f in F.values()
                         for m in sweep(f, DEFAULT_CTX,
                                        prior_ctx="match")],
    ),
    "market_all_methods_sequential": Experiment(
        name="market_all_methods_sequential",
        loader=_market(n_eval=10),
        models=[("prior", {}),                     # yesterday's full surface = delta-CNP's B-spline prior
                ("bspline_data", {}), ("bspline_temporal", {}),
                ("bspline_temporal_graph", {}),   # uniform M=I edges
                ("bspline_tiered_graph", {}),     # SPY-leads-stock tiered precision
                ("bspline_factored_graph", {}),   # + cross-maturity decay
                ("bspline_learned_graph", {}),    # OLS-learned coupling matrix M
                # …same six + within-asset maturity smoothness (λ_maturity>0) to compare
                ("bspline_data_interp", {}), ("bspline_temporal_interp", {}),
                ("bspline_temporal_graph_interp", {}),
                ("bspline_tiered_graph_interp", {}),
                ("bspline_factored_graph_interp", {}),
                ("bspline_learned_graph_interp", {}),
                ("svi_temporal", {}), ("svi_temporal_graph", {}),
                ("ssvi_data", {}),                # per-day SSVI fit (temporal off), takes a long time
                ("ssvi_temporal", {}), ("ssvi_temporal_graph", {}),
                ("kalman_pca", {}), ("kalman_ssvi", {}), ("kalman_ssvi_inc", {}),
                # delta-CNP carries its B-spline prior forward via step(), so it's a
                # temporal model and belongs here; absolute CNP included for reference.
                ("cnp", {"checkpoint": str(TRAINED / "cnp_market.pt")}),
                ("cnp_delta", {"checkpoint": str(TRAINED / "cnp_delta_market.pt")})],
        mode="sequential",
    ),
    # ── leave-one-asset-out: can a model rebuild AAPL with NO AAPL context, from peers? ──
    # Each scores AAPL's targets twice: `ctx_N` (AAPL keeps its context) vs `ctx_N_excl`
    # (peers only).  A small gap = genuine cross-asset predictive power.  Focused on the
    # CNP (cross-asset attention), with prior/ssvi_temporal as references.
    # Focused leave-one-out: predict AAPL's surface with NO AAPL context, purely from the
    # peers + carried prior.  A deliberately SMALL model set so the AAPL-only plots stay
    # legible — the contrast that matters is no-cross-asset vs cross-asset:
    #   prior                 — yesterday's full AAPL surface (persistence floor, no today info)
    #   bspline_temporal      — AAPL carried forward, NO cross-asset coupling (the baseline)
    #   bspline_temporal_graph— + learned cross-asset graph (peers nudge AAPL)
    #   bspline_market_graph  — + single SPY market-factor coupling (the asymmetric-liquidity exploit)
    #   cnp                   — cross-asset attention (learned)
    "market_exclude_aapl": Experiment(
        name="market_exclude_aapl",
        loader=_market(n_eval=10),
        models=[("prior", {}),
                ("bspline_temporal", {}),
                ("bspline_temporal_graph", {}),
                ("bspline_market_graph", {}),
                ("cnp", {"checkpoint": str(TRAINED / "cnp_market.pt")})],
        mode="independent",
        needs_prior=True,
        exclude_asset="AAPL",
    ),
    # Asymmetric liquidity: peers FULL context, AAPL gets the swept quote count. Score AAPL
    # only, so the x-axis is AAPL's OWN number of quotes. Cross-asset models should already
    # reconstruct AAPL well at 2-3 quotes (own quotes pin the level, peers pin the move).
    #
    # prior_ctx="match": AAPL's carried prior is ALSO sparse (its nc quotes from yesterday), so
    #   today's information must do the work — the real test (a full yesterday prior would let
    #   persistence dominate and mask any cross-asset value).
    # Models kept deliberately minimal: prior (persistence floor), bspline_temporal (AAPL's own
    #   data only, no cross-asset), bspline_market_graph (SPY market-factor coupling).
    # lambda_graph=3.0 (not the 0.5 default): the market edge penalises only the LEVEL direction
    #   with precision w=|corr|~0.36, so the default 0.5 puts just lambda_graph*w~0.18 against the
    #   temporal anchor's 1.0 — a ~15% pull, numerically a no-op. lambda_graph~1/E[w]~3 makes the
    #   SPY coupling comparable to the temporal term (so a data-less AAPL node genuinely follows
    #   beta*SPY's move), which is what "the lambdas make sense sizewise" requires.
    "market_aapl_asymmetric": Experiment(
        name="market_aapl_asymmetric",
        loader=_market(n_eval=10),
        # graph_fallback=True: fill each asset's prior across the FULL maturity grid (nearest-
        #   maturity anchor) so data-less target maturities are modelled and the cross-asset edge
        #   can fire on them. All graph models get it, so they differ ONLY in the edge.
        # These edges couple the FULL coefficient (smile-shape) increment, not just ATM:
        #   pca    — rank-2 (denoised) learned map of the increment modes across all peers
        #   learned— full 13x13 OLS map (richer but overfits a short history -> M~0)
        # market (level-only, rank-1) is included as the degenerate reference.
        models=[("prior", {}),
                ("bspline_temporal", {"graph_fallback": True}),
                ("bspline_pca_graph", {"graph_fallback": True}),
                ("bspline_learned_graph", {"graph_fallback": True}),
                ("bspline_market_graph", {"lambda_graph": 3.0, "graph_fallback": True})],
        mode="independent",
        ctx_sizes=ASYM_CTX,
        needs_prior=True,
        asymmetric_target="AAPL",
        prior_ctx="match",
        specs=lambda F: [m for f in F.values()
                         for m in sweep_asymmetric(f, "AAPL", ASYM_CTX,
                                                   prior_ctx="match")],
    ),
    "market_exclude_aapl_sequential": Experiment(
        name="market_exclude_aapl_sequential",
        loader=_market(n_eval=10),
        models=[("prior", {}),
                ("bspline_temporal", {}),
                ("bspline_temporal_graph", {}),
                ("bspline_market_graph", {}),
                ("cnp", {"checkpoint": str(TRAINED / "cnp_market.pt")})],
        mode="sequential",
        needs_prior=True,
        exclude_asset="AAPL",
    ),

    # ══════════════════════════════════════════════════════════════════════════════════
    # DISSERTATION configs — the final runs.
    #
    # Split: the last 900 trading days, 800 train / 100 validation (the eval window where the
    # models actually free-run).  Every model free-runs (prior_mode="carry"): day 0 is seeded
    # from yesterday's full surface, then each day's prior is the model's OWN previous fit.
    #
    # Model set = 3 baselines + 3 cross/non-cross PAIRS, so every cross-asset claim is a clean
    # ablation (the pair differs ONLY in the cross-asset mechanism):
    #   B-spline : temporal-only   vs  + learned 13×13 cross-asset graph
    #   Kalman   : block-diagonal A vs  + full cross-asset A   (kalman_ssvi cross_asset flag)
    #   CNP      : per-asset attn   vs  + joint cross-asset attn (same weights, masked context)
    # Structured models use the full 800-day history (n_history) for their learned coupling.
    # The delta-CNP weights are shared by both CNP variants — train them ONCE into the path
    # below (experiments.train_thesis_cnp), then every config just loads them.
    # ══════════════════════════════════════════════════════════════════════════════════
    THESIS_N_VAL = 100
    THESIS_N_TOTAL = 900
    THESIS_HIST = 800
    CNP_DELTA_CKPT = str(TRAINED / "cnp_delta_thesis.pt")   # increment CNP (perfect-prior ref)
    CNP_ABS_CKPT = str(TRAINED / "cnp_thesis.pt")           # absolute CNP (free-run / sequential)
    THESIS_CTX = (2, 5, 10, 20, 50, 100, 300, 500)   # uniform/extrap sweep (per asset)
    THESIS_ASYM_CTX = (1, 2, 3, 5, 10, 20, 50, 100)  # target's own quote count (peers full)

    def _market_thesis(n_eval=THESIS_N_VAL, n_total=THESIS_N_TOTAL):
        """Last `n_total` trading days, split so the last `n_eval` are the validation window."""
        def load():
            ds = load_grouptech(MARKET_CSV, n_eval_days=n_eval)
            if ds.n_days > n_total:
                ds = ds.subset(list(range(ds.n_days - n_total, ds.n_days)))
            return ds, None
        return load

    # cross / non-cross pairs (registry_name, kwargs) — model.name auto-disambiguates the twins
    # (the `_nox` suffix marks the no-cross-asset variant: block-diagonal A / per-asset attention).
    _PAIR_BSPLINE = [("bspline_temporal_interp",      {}),                          # no graph
                     ("bspline_learned_graph_interp", {"n_history": THESIS_HIST})]  # learned graph
    _PAIR_KAL     = [("kalman_ssvi",     {"cross_asset": False, "n_history": THESIS_HIST}),
                     ("kalman_ssvi",     {"n_history": THESIS_HIST})]               # full cross-asset A
    _PAIR_KAL_INC = [("kalman_ssvi_inc", {"cross_asset": False, "n_history": THESIS_HIST}),
                     ("kalman_ssvi_inc", {"n_history": THESIS_HIST})]               # increment coupling
    _PAIR_CNP     = [("cnp",       {"per_asset": True, "checkpoint": CNP_ABS_CKPT}),   # cnp_nox
                     ("cnp",       {"checkpoint": CNP_ABS_CKPT})]                      # joint attention
    _PAIR_CNP_DLT = [("cnp_delta", {"per_asset": True, "checkpoint": CNP_DELTA_CKPT}), # cnp_delta_nox
                     ("cnp_delta", {"checkpoint": CNP_DELTA_CKPT})]

    # structured models shared by every config (ssvi_temporal + the four cross/non-cross pairs)
    _STRUCT = ([("ssvi_temporal", {})] + _PAIR_BSPLINE + _PAIR_KAL + _PAIR_KAL_INC + _PAIR_CNP)

    # free-run sequential: absolute `cnp` (carries nothing forward → no prior-drift blow-up);
    # `cnp_delta` is EXCLUDED here because its prior compounds under free-run.
    _THESIS_SEQ_MODELS = [("prior", {}), ("bspline_data", {}), ("ssvi_data", {})] + _STRUCT
    # asymmetric target-only: keep `prior` as the "just use yesterday" floor (target still gets nc).
    _THESIS_TGT_MODELS = [("prior", {})] + [("ssvi_temporal", {})] + _PAIR_BSPLINE + _PAIR_KAL + _PAIR_KAL_INC + _PAIR_CNP
    # cold-start (exclude): DROP `prior` — reseed_each_step refreshes it from the target's TRUE
    # previous surface daily, so it never loses the target and unfairly beats the free-runners.
    _THESIS_EXCL_MODELS = [("ssvi_temporal", {})] + _PAIR_BSPLINE + _PAIR_KAL + _PAIR_KAL_INC + _PAIR_CNP
    # perfect-prior reference (re-anchored daily): include BOTH CNP families — `cnp_delta` is
    # stable and genuinely useful here because the prior it subtracts is accurate every day.
    _THESIS_PP_MODELS = [("prior", {}), ("bspline_data", {}), ("ssvi_data", {})] + _STRUCT + _PAIR_CNP_DLT

    def _sweep_carry(fitter, ctx_sizes):
        """Free-run Models for every (regime, ctx): score the full surface, carry own fit forward."""
        from surfacelab.eval import Model, Uniform, Extrap, Full
        # persistence (PriorModel) must re-seed from the true previous day every step; structured
        # models carry their own fit (reseed_each_step stays False).
        reseed = getattr(fitter, "reseed_each_step", False)
        return [Model(fitter=fitter,
                      today=Extrap(nc) if reg == "extrap" else Uniform(nc),
                      yesterday=Full(), prior_mode="carry", reseed_each_step=reseed)
                for nc in ctx_sizes for reg in ("unif", "extrap")]

    def _sweep_target_carry(fitter, target, ctx_sizes, splitter):
        """Free-run Models for a single target asset using `splitter` (Exclude or Asymmetric):
        peers observed, the target gets `nc` (Asymmetric) or zero (Exclude) of its own context;
        only the target is scored; the model carries its own fit forward."""
        from surfacelab.eval import Model, Full
        reseed = getattr(fitter, "reseed_each_step", False)
        return [Model(fitter=fitter, today=splitter(target, nc, regime=reg),
                      yesterday=Full(), prior_mode="carry", reseed_each_step=reseed)
                for nc in ctx_sizes for reg in ("unif", "extrap")]

    def _thesis_exclude(target):
        from surfacelab.eval import Exclude
        return Experiment(
            name=f"thesis_exclude_{target.lower()}",
            loader=_market_thesis(), models=_THESIS_EXCL_MODELS, mode="sequential",
            needs_prior=True, exclude_asset=target,
            specs=lambda F: [m for f in F.values()
                             for m in _sweep_target_carry(f, target, THESIS_CTX, Exclude)],
        )

    def _thesis_asym(target):
        from surfacelab.eval import Asymmetric
        return Experiment(
            name=f"thesis_asym_{target.lower()}",
            loader=_market_thesis(), models=_THESIS_TGT_MODELS, mode="sequential",
            needs_prior=True, asymmetric_target=target,
            specs=lambda F: [m for f in F.values()
                             for m in _sweep_target_carry(f, target, THESIS_ASYM_CTX, Asymmetric)],
        )

    "thesis_sequential": Experiment(
        name="thesis_sequential",
        loader=_market_thesis(), models=_THESIS_SEQ_MODELS, mode="sequential",
        needs_prior=True,
        specs=lambda F: [m for f in F.values() for m in _sweep_carry(f, THESIS_CTX)],
    ),
    # (2) cold-start: the target asset gets ZERO context; rebuilt purely from peers + carry.
    "thesis_exclude_aapl":  _thesis_exclude("AAPL"),
    "thesis_exclude_googl": _thesis_exclude("GOOGL"),
    # (3) graceful degradation: peers FULL, the target gets a swept number of its own quotes.
    "thesis_asym_aapl":  _thesis_asym("AAPL"),
    "thesis_asym_googl": _thesis_asym("GOOGL"),
    # (4) perfect-prior reference (informative, not realistic-free-run): re-anchor every model's
    # prior to yesterday's FULL true surface each day (prior_mode="fit", yesterday=Full), so no
    # error compounds.  cnp_delta is stable here and should shine at high context where the
    # B-spline prior it subtracts is already accurate.
    "thesis_perfect_prior": Experiment(
        name="thesis_perfect_prior",
        loader=_market_thesis(), models=_THESIS_PP_MODELS, mode="independent",
        needs_prior=True,
        specs=lambda F: [m for f in F.values()
                         for m in sweep(f, THESIS_CTX, prior_ctx="full", prior_mode="fit")],
    ),
    # (5) Kalman-on-increments across representations: is SSVI the ceiling-limiter?  Same
    # free-run setup as thesis_sequential, comparing the increment trick on SSVI params vs
    # richer linear bases (PCA factors, B-spline coefficients), with cross/nox where defined.
    "thesis_kalman_coeff": Experiment(
        name="thesis_kalman_coeff",
        loader=_market_thesis(), mode="sequential", needs_prior=True,
        models=[("prior", {}), ("bspline_data", {}),
                ("kalman_ssvi_inc", {"n_history": THESIS_HIST}),                       # SSVI increment baseline
                ("kalman_ssvi_inc", {"cross_asset": False, "n_history": THESIS_HIST}),  # → kalman_ssvi_inc_nox
                ("kalman_pca", {}), ("kalman_pca_inc", {}),                            # PCA levels vs increments
                ("kalman_bspline", {"n_history": THESIS_HIST}),                        # B-spline levels
                ("kalman_bspline_nox", {"n_history": THESIS_HIST}),
                ("kalman_bspline_inc", {"n_history": THESIS_HIST}),                    # B-spline increments
                ("kalman_bspline_inc_nox", {"n_history": THESIS_HIST})],
        specs=lambda F: [m for f in F.values() for m in _sweep_carry(f, THESIS_CTX)],
    ),

    # ── New experiments added for iv_surface_mt0856ij_1/2 and mt08kvrq_1/2 ──────────
    # These use the standard _heston() and _market() loaders with default parameters.
    # The model lists are placeholders; they must correspond to actual registered model
    # classes in the codebase.
    "iv_surface_mt0856ij_1": Experiment(
        name="iv_surface_mt0856ij_1",
        loader=_heston(),  # surfacelab/data/heston.py
        models=[
            ("model", {}),
            ("bspline_basis", {}),
            ("module", {}),
            ("trainer", {}),
            ("edges", {}),
            ("factors", {}),
            ("kalman_ssvi", {}),
            ("kalman", {}),
            ("base", {}),
            ("pca", {}),
            ("representations", {}),
            ("prior_baseline", {}),
            ("registry", {}),
            ("regularized", {}),
        ],
        mode="independent",
    ),
    "iv_surface_mt0856ij_2": Experiment(
        name="iv_surface_mt0856ij_2",
        loader=_market(),  # surfacelab/data/market.py
        models=[
            ("model", {}),
            ("bspline_basis", {}),
            ("module", {}),
            ("trainer", {}),
            ("edges", {}),
            ("factors", {}),
            ("kalman_ssvi", {}),
            ("kalman", {}),
            ("base", {}),
            ("pca", {}),
            ("representations", {}),
            ("prior_baseline", {}),
            ("registry", {}),
            ("regularized", {}),
        ],
        mode="independent",
    ),
    "iv_surface_mt08kvrq_1": Experiment(
        name="iv_surface_mt08kvrq_1",
        loader=_heston(),  # surfacelab/data/heston.py
        models=[
            ("bspline_basis", {}),
            ("module", {}),
            ("model", {}),
            ("trainer", {}),
            ("edges", {}),
            ("factors", {}),
            ("kalman_ssvi", {}),
            ("kalman", {}),
            ("base", {}),
            ("pca", {}),
            ("representations", {}),
            ("prior_baseline", {}),
            ("registry", {}),
            ("regularized", {}),
        ],
        mode="independent",
    ),
    "iv_surface_mt08kvrq_2": Experiment(
        name="iv_surface_mt08kvrq_2",
        loader=_market(),  # surfacelab/data/market.py
        models=[
            ("bspline_basis", {}),
            ("module", {}),
            ("model", {}),
            ("trainer", {}),
            ("edges", {}),
            ("factors", {}),
            ("kalman_ssvi", {}),
            ("kalman", {}),
            ("base", {}),
            ("pca", {}),
            ("representations", {}),
            ("prior_baseline", {}),
            ("registry", {}),
            ("regularized", {}),
        ],
        mode="independent",
    ),
}


def get_experiment(name: str) -> Experiment:
    if name not in EXPERIMENTS:
        raise KeyError(f"unknown experiment '{name}'. Known: {sorted(EXPERIMENTS)}")
    return EXPERIMENTS[name]
