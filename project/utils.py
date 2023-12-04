import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

feature_short_names = [
    "Name",
    "Alpha-2",
    "Year",
    "Electricity Access (%)",
    "Growth in Income per Capita (%)",
    "Net Savings (% GNI)",
    "Carbon Dioxide Damage (% GNI)",
    "Natural Resources Depletion (% GNI)",
    "Net Forest Depletion (% GNI)",
    "Particulate Emissions Damage (% GNI)",
    "ATMs (per 100K adults)",
    "Broad Money (% GDP)",
    "Children out of School (%)",
    "Years Compulsory Education",
    "Business Start-up Costs (Female, % GNI)",
    "Business Start-up Costs (Male, % GNI)",
    "Exports (% GDP)",
    "Consumption Expenditure (% GDP)",
    "GDP ($)",
    "GDP per Capita ($)",
    "Government Consumption Expenditure (% GDP)",
    "Gross National Expenditure (% GDP)",
    "Gross Savings (% GDP)",
    "Imports (% GDP)",
    "Annual Inflation (%)",
    "Primary School Completion (%)",
    "Females in Natl. Parliaments (%)",
    "Pupil-teacher Ratio",
    "Renewable Energy Output (% total)",
    "Renewable Energy Consumption (% total)",
    "Pre-primary School Enrollment (%)",
    "Primary School Enrollment (%)",
    "Secondary School Enrollment (%)",
    "Trade (% GDP)",
    "Women Business and Law Index",
    "Undernourishment (%)",
    "Below Internatl. Poverty Line (%)",
    "Population Covered by 2G+ Network (%)",
    "Population Covered by 3G+ Network (%)",
    "Access to Drinking Water (%)",
    "Male Unemployment Rate (%)",
    "Female Unemployment Rate (%)",
    "CO2 Emissions",
    "Gini Index",
    "Population Using Internet (%)",
    "Life Expectancy",
    "Total Population",
    "Regime Type",
    "Rural Population (%)",
    "Natural Resource Rents (% GDP)",
    "Urban Population (%)",
]


def create_pca(df):
    df_std = StandardScaler().fit_transform(df)

    pca = PCA()
    components = pca.fit_transform(df_std)

    return pca, pd.DataFrame(data=components)


def viz_pca(pca, extra_title=None):
    sns.set_theme(style="whitegrid")

    explained_variance = pca.explained_variance_ratio_
    cumulative_explained_variance = np.cumsum(explained_variance)

    plt.figure(figsize=(12, 6))
    sns.barplot(
        x=np.arange(1, len(explained_variance) + 1),
        y=explained_variance,
        color="skyblue",
        alpha=0.7,
        label="Individual explained variance",
    )

    sns.lineplot(
        x=np.arange(1, len(cumulative_explained_variance) + 1),
        y=cumulative_explained_variance,
        marker="o",
        color="red",
        label="Cumulative explained variance",
    ).set_title(f"PCA {extra_title}", fontsize=15, weight="bold")

    plt.axhline(
        y=0.95, color="r", linestyle="--", linewidth=2, label="95% Explained Variance"
    )

    threshold_index = np.where(cumulative_explained_variance >= 0.95)[0][0] + 1
    plt.axvline(x=threshold_index, color="green", linestyle="--", linewidth=2)

    plt.ylabel("Explained variance")
    plt.xlabel("Principal components")

    plt.xticks(range(1, len(explained_variance) + 1, 5))

    plt.legend(loc="best")
    plt.tight_layout()
    return plt


def create_loadings(df, pca):
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

    return pd.DataFrame(
        loadings,
        index=df.columns,
        columns=[f"PC{i+1}" for i in range(loadings.shape[1])],
    )


def print_top_features_for_component(pca, loadings, n):
    for i in range(len(pca.components_)):
        explained_var = loadings.iloc[:, i].pow(2)
        top_features = explained_var.sort_values(ascending=False)[:n]

        print(f"PC{i+1} top {n} features and explained variance:")
        for feature in top_features.index:
            print(f"{feature}: {top_features[feature]:.4f}")
        print("")


def viz_pca_heatmap(df, n):
    scaler = StandardScaler()

    df_std = scaler.fit_transform(df)

    pca = PCA(n_components=n)
    pca_df = pd.DataFrame(pca.fit_transform(df_std))

    loadings = create_loadings(df, pca)

    plt.figure(figsize=(12, 8))
    sns.heatmap(loadings, annot=False, cmap="coolwarm", center=0)
    plt.title("PCA Loadings")
    plt.xlabel("Principal Components")
    plt.ylabel("Original Features")
    return pca, pca_df, plt


def viz_corr_heatmap(corr_df, is_absolute):
    title_str = "Correlation Between Features"
    if is_absolute:
        title_str = "Correlation (Absolute) Between Features"

    mask = np.triu(np.ones_like(corr_df, dtype=bool))

    f, ax = plt.subplots(figsize=(11, 9))

    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    sns.heatmap(
        corr_df,
        mask=mask,
        cmap=cmap,
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.5},
    ).set_title(title_str, fontsize=15, weight="bold")
    ax.set_xticks(np.arange(corr_df.shape[1]))
    ax.set_xticklabels(corr_df.columns, rotation=90)

    return f


def viz_regression(actual, predicted, y_test):
    sns.set_theme(style="whitegrid")
    results = pd.DataFrame({"Actual": actual, "Predicted": predicted})
    plt.figure(figsize=(10, 6))

    plot = sns.scatterplot(x="Actual", y="Predicted", data=results)
    plt.plot(
        [y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color="red", lw=2
    )  # Diagonal line
    plt.title("Actual vs Predicted Values (Linear Regression)")
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    return plot
