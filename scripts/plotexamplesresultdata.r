library(tidyverse)

# ---------------------------------------------------------------------------
# residual_vs_plain.csv -- training-loss curves (plain vs residual)
# ---------------------------------------------------------------------------

losses <- readr::read_csv("residual_vs_plain.csv")

p_loss <- losses |>
    tidyr::pivot_longer(c(plain_mse, residual_mse), names_to = "model", values_to = "mse") |>
    dplyr::mutate(model = dplyr::recode(model, plain_mse = "plain", residual_mse = "residual")) |>
    ggplot(aes(x = epoch, y = mse, color = model)) +
    geom_line(linewidth = 0.9) +
    scale_y_log10() +
    labs(
        title = "Deep MLP training loss: plain vs residual",
        x = "epoch", y = "training MSE (log)", color = NULL
    ) +
    theme_minimal()

print(p_loss)

# ---------------------------------------------------------------------------
# residual_ensemble_uncertainty.csv -- ensemble predictions + uncertainty
# ---------------------------------------------------------------------------

ens <- readr::read_csv("residual_ensemble_uncertainty.csv")

# Training boundary inferred from is_ood transitions in the data.
train_min <- min(ens$x[ens$is_ood == 0])
train_max <- max(ens$x[ens$is_ood == 0])

# Long form for the per-member lines.
member_cols <- grep("^m[0-9]+$", names(ens), value = TRUE)
members_long <- ens |>
    dplyr::select(x, all_of(member_cols)) |>
    tidyr::pivot_longer(-x, names_to = "member", values_to = "y")

p_unc <- ggplot(ens, aes(x = x)) +
    # Shade the OOD regions so the holdout is obvious at a glance.
    annotate("rect",
        xmin = -Inf, xmax = train_min, ymin = -Inf, ymax = Inf,
        fill = "grey90", alpha = 0.7
    ) +
    annotate("rect",
        xmin = train_max, xmax = Inf, ymin = -Inf, ymax = Inf,
        fill = "grey90", alpha = 0.7
    ) +
    # Mean ± std uncertainty band.
    geom_ribbon(aes(ymin = mean - std, ymax = mean + std),
        fill = "#1f78b4", alpha = 0.25
    ) +
    # Each ensemble member as a faint line.
    geom_line(
        data = members_long, aes(x = x, y = y, group = member),
        color = "#1f78b4", alpha = 0.35, linewidth = 0.4
    ) +
    # Ensemble mean prediction.
    geom_line(aes(y = mean), color = "#1f78b4", linewidth = 1.0) +
    # Ground truth (no noise).
    geom_line(aes(y = truth),
        color = "black", linewidth = 0.8, linetype = "dashed"
    ) +
    # Mark the training boundary explicitly.
    geom_vline(
        xintercept = c(train_min, train_max),
        color = "grey40", linetype = "dotted"
    ) +
    annotate("text",
        x = (train_min + train_max) / 2, y = Inf,
        label = "training range", vjust = 1.5, color = "grey30", size = 3
    ) +
    annotate("text",
        x = train_min - 1.5, y = Inf,
        label = "OOD", vjust = 1.5, color = "grey30", size = 3
    ) +
    annotate("text",
        x = train_max + 1.5, y = Inf,
        label = "OOD", vjust = 1.5, color = "grey30", size = 3
    ) +
    labs(
        title = "Residual ensemble: prediction + uncertainty",
        subtitle = sprintf(
            "trained on x in [%g, %g]; %d members; band = mean ± std",
            train_min, train_max, length(member_cols)
        ),
        x = "x", y = "y"
    ) +
    theme_minimal()

print(p_unc)
