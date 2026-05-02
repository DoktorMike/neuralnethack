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
# Ensemble uncertainty plots: same layout, applied to whichever CSVs exist.
# ---------------------------------------------------------------------------

plot_uncertainty <- function(csv_path, title) {
    if (!file.exists(csv_path)) {
        message("skip: ", csv_path, " not found")
        return(invisible(NULL))
    }
    ens <- readr::read_csv(csv_path)

    # Training boundary inferred from is_ood transitions in the data.
    train_min <- min(ens$x[ens$is_ood == 0])
    train_max <- max(ens$x[ens$is_ood == 0])

    # Long form for the per-member lines.
    member_cols <- grep("^m[0-9]+$", names(ens), value = TRUE)
    members_long <- ens |>
        dplyr::select(x, all_of(member_cols)) |>
        tidyr::pivot_longer(-x, names_to = "member", values_to = "y")

    p <- ggplot(ens, aes(x = x)) +
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
            x = train_min - (train_max - train_min) / 4, y = Inf,
            label = "OOD", vjust = 1.5, color = "grey30", size = 3
        ) +
        annotate("text",
            x = train_max + (train_max - train_min) / 4, y = Inf,
            label = "OOD", vjust = 1.5, color = "grey30", size = 3
        ) +
        labs(
            title = title,
            subtitle = sprintf(
                "trained on x in [%g, %g]; %d members; band = mean ± std",
                train_min, train_max, length(member_cols)
            ),
            x = "x", y = "y"
        ) +
        theme_minimal()

    print(p)
}

plot_uncertainty(
    "residual_ensemble_uncertainty.csv",
    "Residual ensemble: y = x + 0.3 sin(5x)"
)

plot_uncertainty(
    "cubic_ensemble_uncertainty.csv",
    "Residual ensemble: Amini cubic y = x^3 (noise sd = 3)"
)

# ---------------------------------------------------------------------------
# Generic 3-class ensemble uncertainty plot. Renders the decision surface
# coloured by mean softmax probabilities (R, G, B = classes 0, 1, 2) with
# intensity scaled by 1 - normalised entropy. The CSVs follow the layout
# emitted by examples/iris_ensemble_uncertainty and friends.
#
# Three views via the Depeweg et al. 2018 decomposition:
#   total      = H(p̄)              -- conflated total uncertainty
#   aleatoric  = mean_i H(p_i)     -- ambiguity all members agree on
#   epistemic  = total - aleatoric -- member disagreement / OOD-ness (BALD)
# ---------------------------------------------------------------------------

plot_3class_uncertainty <- function(entropy_col,
                                    title,
                                    grid_path,
                                    obs_path,
                                    class_labels = c("class 0", "class 1", "class 2"),
                                    xlab = "x1", ylab = "x2") {
    if (!file.exists(grid_path) || !file.exists(obs_path)) {
        message("skip: ", grid_path, " or ", obs_path, " not found")
        return(invisible(NULL))
    }
    grid <- readr::read_csv(grid_path, show_col_types = FALSE)
    obs <- readr::read_csv(obs_path, show_col_types = FALSE)
    if (!entropy_col %in% names(grid)) {
        message("skip: column '", entropy_col, "' missing in ", grid_path)
        return(invisible(NULL))
    }

    # Intensity: 1 = certain (entropy = 0), 0 = max entropy = log K.
    # Total/aleatoric are bounded by log(K). Epistemic is also bounded by
    # log(K) but in practice rarely fills that range, so it'll look paler
    # than the total view -- that is faithful, not a normalisation bug.
    K <- 3
    grid$h <- grid[[entropy_col]]
    grid <- grid |>
        dplyr::mutate(
            intensity = pmax(0, pmin(1, 1 - h / log(K))),
            fill_color = rgb(p0 * intensity, p1 * intensity, p2 * intensity)
        )

    # Training extent (from non-OOD grid points, equivalently the obs
    # range). Used to draw the in-distribution boundary.
    trn <- obs |> dplyr::filter(set == "trn")
    train_box <- list(
        x1_min = min(trn$x1), x1_max = max(trn$x1),
        x2_min = min(trn$x2), x2_max = max(trn$x2)
    )

    obs <- obs |>
        dplyr::mutate(
            true_class = factor(true_class, levels = 0:2, labels = class_labels),
            misclass = true_class != factor(pred_class, levels = 0:2, labels = class_labels)
        )
    shape_map <- setNames(c(21, 22, 24), class_labels)

    subtitle <- sprintf(
        "RGB = mean softmax (R=%s, G=%s, B=%s); intensity = 1 - %s/log(%d)",
        class_labels[1], class_labels[2], class_labels[3], entropy_col, K
    )

    p <- ggplot() +
        geom_tile(data = grid, aes(x = x1, y = x2, fill = fill_color)) +
        scale_fill_identity() +
        # Training extent rectangle.
        annotate("rect",
            xmin = train_box$x1_min, xmax = train_box$x1_max,
            ymin = train_box$x2_min, ymax = train_box$x2_max,
            fill = NA, color = "white", linetype = "dashed", linewidth = 0.4
        ) +
        # Observations: shape by true class, white border to read against
        # any background colour. Misclassified points get a red ring.
        geom_point(
            data = obs, aes(x = x1, y = x2, shape = true_class),
            color = "white", fill = "black", size = 2, stroke = 0.6
        ) +
        geom_point(
            data = obs |> dplyr::filter(misclass),
            aes(x = x1, y = x2),
            shape = 21, color = "red", fill = NA, size = 4, stroke = 0.8
        ) +
        scale_shape_manual(values = shape_map) +
        coord_equal(expand = FALSE) +
        labs(
            title = title, subtitle = subtitle,
            x = xlab, y = ylab, shape = "true class",
            caption = "white dashed = training extent; red ring = misclassified"
        ) +
        theme_minimal() +
        theme(panel.grid = element_blank())

    print(p)
}

# Iris (petal length / width).
plot_3class_uncertainty("entropy_total",
    "Iris ensemble uncertainty -- total (H of mean softmax)",
    "iris_uncertainty_grid.csv", "iris_uncertainty_obs.csv",
    class_labels = c("setosa", "versicolour", "virginica"),
    xlab = "petal length (z-norm)", ylab = "petal width (z-norm)"
)
plot_3class_uncertainty("entropy_aleatoric",
    "Iris ensemble uncertainty -- aleatoric (mean of per-member H)",
    "iris_uncertainty_grid.csv", "iris_uncertainty_obs.csv",
    class_labels = c("setosa", "versicolour", "virginica"),
    xlab = "petal length (z-norm)", ylab = "petal width (z-norm)"
)
plot_3class_uncertainty("entropy_epistemic",
    "Iris ensemble uncertainty -- epistemic (BALD: total - aleatoric)",
    "iris_uncertainty_grid.csv", "iris_uncertainty_obs.csv",
    class_labels = c("setosa", "versicolour", "virginica"),
    xlab = "petal length (z-norm)", ylab = "petal width (z-norm)"
)

# 3-arm Archimedean spiral.
plot_3class_uncertainty("entropy_total",
    "Spiral ensemble uncertainty -- total (H of mean softmax)",
    "spiral_uncertainty_grid.csv", "spiral_uncertainty_obs.csv",
    class_labels = c("arm 0", "arm 1", "arm 2"),
    xlab = "x1 (z-norm)", ylab = "x2 (z-norm)"
)

plot_3class_uncertainty("entropy_aleatoric",
    "Spiral ensemble uncertainty -- aleatoric (mean of per-member H)",
    "spiral_uncertainty_grid.csv", "spiral_uncertainty_obs.csv",
    class_labels = c("arm 0", "arm 1", "arm 2"),
    xlab = "x1 (z-norm)", ylab = "x2 (z-norm)"
)

plot_3class_uncertainty("entropy_epistemic",
    "Spiral ensemble uncertainty -- epistemic (BALD: total - aleatoric)",
    "spiral_uncertainty_grid.csv", "spiral_uncertainty_obs.csv",
    class_labels = c("arm 0", "arm 1", "arm 2"),
    xlab = "x1 (z-norm)", ylab = "x2 (z-norm)"
)
