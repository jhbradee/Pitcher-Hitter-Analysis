import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings

warnings.filterwarnings("ignore")

# set plotting style
plt.style.use("seaborn-v0_8")
sns.set_palette("husl")

############################################################################
# make sure data files are loaded
############################################################################


def load_data():
    """Load all processed data files"""

    data = {}

    try:
        data["pitcher_overall"] = pd.read_parquet(
            "data/processed/pitcher_overall_summary.parquet"
        )
        data["pitcher_by_pitch"] = pd.read_parquet(
            "data/processed/pitcher_profile_by_pitch_type.parquet"
        )
        data["batter_overall"] = pd.read_parquet(
            "data/processed/batter_overall_summary.parquet"
        )
        data["batter_by_pitch"] = pd.read_parquet(
            "data/processed/batter_by_pitch_type_profiles.parquet"
        )
        data["batter_vs_handedness"] = pd.read_parquet(
            "data/processed/batter_vs_pitcher_handedness.parquet"
        )
        data["pitch_level"] = pd.read_parquet(
            "data/processed/pitch_level_enhanced.parquet"
        )

        print("Data files loaded successfully.")
        for key, df in data.items():
            print(f"{key}: {df.shape}")
    except FileNotFoundError as e:
        print(f"Error loading data files: {e}")
        print("Make sure you've run the data processing pipeline first.")

    return data


############################################################################
# PITCHER PERFORMANCE VISUALIZATIONS
############################################################################


def plot_pitcher_velocity_heatmap(pitcher_by_pitch, top_n=20):
    """Plot heatmap of average pitch velocities for top N pitchers by total pitches thrown"""

    # Get top N pitchers by total pitches thrown
    top_pitchers = (
        pitcher_by_pitch.groupby("player_name")["total_pitches"]
        .sum()
        .nlargest(top_n)
        .index
    )

    # filter and pivot data
    viz_data = pitcher_by_pitch[pitcher_by_pitch["player_name"].isin(top_pitchers)]

    # create pivot table
    heatmap_data = viz_data.pivot_table(
        index="player_name",
        columns="pitch_name",
        values="release_speed_mean",
        fill_value=0,
    )

    # create heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt=".1f",
        cmap="RdYlBu_r",
        cbar_kws={"label": "Avg Velocity (mph)"},
    )

    plt.title(f"Pitcher Velocity by Pitch Type - Top {top_n} Pitchers")
    plt.xlabel("Pitch Type")
    plt.ylabel("Pitcher")
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

    return plt.gcf()


def plot_stuff_vs_command_scatter(pitcher_overall):
    """
    Scatter plot: Stuff vs Command with pitcher classifications
    """
    fig = px.scatter(
        pitcher_overall,
        x="usage_weighted_command",
        y="usage_weighted_stuff",
        size="total_pitches",
        color="overall_whiff_rate",
        hover_name="formatted_name",
        hover_data=["overall_woba_against", "repertoire_size"],
        title="Pitcher Stuff vs Command Profile",
        labels={
            "usage_weighted_command": "Command Score",
            "usage_weighted_stuff": "Stuff Score",
            "overall_whiff_rate": "Whiff Rate",
        },
        color_continuous_scale="Viridis",
    )

    # Add quadrant lines
    fig.add_hline(
        y=pitcher_overall["usage_weighted_stuff"].median(),
        line_dash="dash",
        line_color="gray",
        opacity=0.5,
    )
    fig.add_vline(
        x=pitcher_overall["usage_weighted_command"].median(),
        line_dash="dash",
        line_color="gray",
        opacity=0.5,
    )

    # Add quadrant labels
    fig.add_annotation(
        x=0.95,
        y=0.95,
        text="Elite<br>Stuff+Command",
        xref="paper",
        yref="paper",
        showarrow=False,
    )
    fig.add_annotation(
        x=0.05,
        y=0.95,
        text="Stuff<br>No Command",
        xref="paper",
        yref="paper",
        showarrow=False,
    )
    fig.add_annotation(
        x=0.95,
        y=0.05,
        text="Command<br>No Stuff",
        xref="paper",
        yref="paper",
        showarrow=False,
    )

    fig.update_layout(height=600)
    return fig


def plot_pitch_arsenal_radar(pitcher_by_pitch, pitcher_name):
    """
    Radar chart for individual pitcher's arsenal
    """
    pitcher_data = pitcher_by_pitch[
        pitcher_by_pitch["player_name"].str.contains(pitcher_name, case=False, na=False)
    ].copy()

    if len(pitcher_data) == 0:
        print(f"No data found for pitcher containing '{pitcher_name}'")
        return None

    # Normalize metrics to 0-100 scale
    metrics = ["csw_rate", "stuff_score", "command_score", "pitch_type_percentage"]

    for metric in metrics:
        if metric in pitcher_data.columns:
            pitcher_data[f"{metric}_norm"] = (
                (pitcher_data[metric] - pitcher_data[metric].min())
                / (pitcher_data[metric].max() - pitcher_data[metric].min())
                * 100
            )

    # Create radar chart
    fig = go.Figure()

    for _, pitch in pitcher_data.iterrows():
        fig.add_trace(
            go.Scatterpolar(
                r=[pitch.get(f"{m}_norm", 0) for m in metrics],
                theta=["CSW Rate", "Stuff Score", "Command Score", "Usage Rate"],
                fill="toself",
                name=pitch["pitch_name"],
            )
        )

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=True,
        title=f"Pitch Arsenal Profile: {pitcher_data.iloc[0]['player_name']}",
    )

    return fig


# =============================================================================
# 2. BATTER PERFORMANCE VISUALIZATIONS
# =============================================================================


def plot_contact_quality_matrix(batter_by_pitch, min_pitches=50):
    """
    Bubble chart: Exit velocity vs Barrel rate by pitch type
    """
    # Filter for sufficient sample size
    viz_data = batter_by_pitch[batter_by_pitch["pitches_seen"] >= min_pitches].copy()

    fig = px.scatter(
        viz_data,
        x="avg_exit_velocity",
        y="barrel_rate",
        size="pitches_seen",
        color="pitch_name",
        hover_name="formatted_name",
        hover_data=["whiff_rate", "chase_rate"],
        title=f"Batter Contact Quality by Pitch Type (Min {min_pitches} pitches)",
        labels={
            "avg_exit_velocity": "Average Exit Velocity (mph)",
            "barrel_rate": "Barrel Rate",
            "pitches_seen": "Pitches Seen",
        },
    )

    fig.update_layout(height=600)
    return fig


def plot_discipline_profiles(batter_overall, min_pitches=200):
    """
    Scatter plot of plate discipline metrics
    """
    viz_data = batter_overall[
        batter_overall["total_pitches_seen"] >= min_pitches
    ].copy()

    fig = px.scatter(
        viz_data,
        x="chase_rate",
        y="zone_swing_rate",
        color="batter_side",
        size="overall_woba",
        hover_name="formatted_name",
        hover_data=["overall_whiff_rate", "discipline_score"],
        title="Batter Plate Discipline Profiles",
        labels={
            "chase_rate": "Chase Rate (Swing at balls)",
            "zone_swing_rate": "Zone Swing Rate",
            "overall_woba": "wOBA",
        },
    )

    # Add reference lines for league averages
    fig.add_hline(
        y=viz_data["zone_swing_rate"].median(),
        line_dash="dash",
        line_color="gray",
        opacity=0.5,
    )
    fig.add_vline(
        x=viz_data["chase_rate"].median(),
        line_dash="dash",
        line_color="gray",
        opacity=0.5,
    )

    return fig


# =============================================================================
# 3. MATCHUP ANALYSIS
# =============================================================================


def plot_platoon_advantage_heatmap(batter_vs_handedness):
    """
    Heatmap showing platoon advantages
    """
    # Extract relevant columns and reshape
    platoon_data = []

    for _, row in batter_vs_handedness.iterrows():
        if "woba_value_vs_L" in row and "woba_value_vs_R" in row:
            platoon_data.append(
                {
                    "batter": row.get("batter", "Unknown"),
                    "vs_LHP": row["woba_value_vs_L"],
                    "vs_RHP": row["woba_value_vs_R"],
                }
            )

    if not platoon_data:
        print("No platoon data available")
        return None

    platoon_df = pd.DataFrame(platoon_data)
    platoon_df["platoon_advantage"] = platoon_df["vs_LHP"] - platoon_df["vs_RHP"]

    # Create summary by handedness matchup
    summary_data = {
        "LHB vs LHP": platoon_df["vs_LHP"].mean(),
        "LHB vs RHP": platoon_df["vs_RHP"].mean(),
        "RHB vs LHP": platoon_df["vs_LHP"].mean(),
        "RHB vs RHP": platoon_df["vs_RHP"].mean(),
    }

    # Reshape for heatmap
    heatmap_data = pd.DataFrame(
        [
            ["LHB", "vs LHP", summary_data["LHB vs LHP"]],
            ["LHB", "vs RHP", summary_data["LHB vs RHP"]],
            ["RHB", "vs LHP", summary_data["RHB vs LHP"]],
            ["RHB", "vs RHP", summary_data["RHB vs RHP"]],
        ],
        columns=["Batter_Hand", "Pitcher_Hand", "wOBA"],
    )

    heatmap_pivot = heatmap_data.pivot("Batter_Hand", "Pitcher_Hand", "wOBA")

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        heatmap_pivot,
        annot=True,
        fmt=".3f",
        cmap="RdYlGn",
        center=0.320,  # League average wOBA
        cbar_kws={"label": "wOBA"},
    )

    plt.title("Platoon Advantage Matrix", fontsize=16)
    plt.xlabel("Pitcher Handedness")
    plt.ylabel("Batter Handedness")

    return plt.gcf()


def plot_velocity_vs_contact(pitcher_overall, batter_overall):
    """
    Scatter plot: Pitcher velocity vs Batter contact ability
    """
    # Create all possible matchups (simplified for visualization)
    sample_pitchers = pitcher_overall.sample(min(50, len(pitcher_overall)))
    sample_batters = batter_overall.sample(min(50, len(batter_overall)))

    fig = px.scatter(
        x=sample_pitchers["avg_velocity"],
        y=sample_batters["weighted_contact_rate"],
        hover_name=[
            f"P: {p} vs B: {b}"
            for p in sample_pitchers["formatted_name"]
            for b in sample_batters["formatted_name"][:1]
        ],
        title="Pitcher Velocity vs Batter Contact Ability",
        labels={"x": "Pitcher Average Velocity (mph)", "y": "Batter Contact Rate"},
    )

    return fig


# =============================================================================
# 4. LEAGUE TRENDS AND ANALYSIS
# =============================================================================


def plot_pitch_mix_evolution(pitcher_by_pitch):
    """
    Bubble chart: Pitch usage vs effectiveness
    """
    # Aggregate by pitch type
    pitch_summary = (
        pitcher_by_pitch.groupby("pitch_name")
        .agg(
            {
                "pitch_type_percentage": "mean",
                "csw_rate": "mean",
                "pitcher": "count",
                "total_pitches": "sum",
            }
        )
        .reset_index()
    )

    pitch_summary.columns = [
        "pitch_name",
        "avg_usage_pct",
        "avg_csw_rate",
        "num_pitchers",
        "total_pitches",
    ]

    fig = px.scatter(
        pitch_summary,
        x="avg_usage_pct",
        y="avg_csw_rate",
        size="total_pitches",
        color="num_pitchers",
        hover_name="pitch_name",
        title="League Pitch Mix: Usage vs Effectiveness",
        labels={
            "avg_usage_pct": "Average Usage %",
            "avg_csw_rate": "Average CSW Rate",
            "total_pitches": "Total Pitches",
            "num_pitchers": "Pitchers Throwing",
        },
    )

    return fig


def plot_power_vs_contact_spectrum(batter_overall, min_pitches=200):
    """
    Quadrant analysis: Power vs Contact
    """
    viz_data = batter_overall[
        batter_overall["total_pitches_seen"] >= min_pitches
    ].copy()

    fig = px.scatter(
        viz_data,
        x="contact_quality_score",
        y="power_score",
        color="batter_side",
        size="overall_woba",
        hover_name="formatted_name",
        hover_data=["avg_exit_velocity", "barrel_rate"],
        title="Batter Profiles: Power vs Contact Quality",
        labels={
            "contact_quality_score": "Contact Quality Score",
            "power_score": "Power Score",
        },
    )

    # Add quadrant lines
    fig.add_hline(
        y=viz_data["power_score"].median(),
        line_dash="dash",
        line_color="gray",
        opacity=0.5,
    )
    fig.add_vline(
        x=viz_data["contact_quality_score"].median(),
        line_dash="dash",
        line_color="gray",
        opacity=0.5,
    )

    # Add quadrant labels
    fig.add_annotation(
        x=0.95,
        y=0.95,
        text="Power +<br>Contact",
        xref="paper",
        yref="paper",
        showarrow=False,
    )
    fig.add_annotation(
        x=0.05,
        y=0.95,
        text="Power<br>No Contact",
        xref="paper",
        yref="paper",
        showarrow=False,
    )
    fig.add_annotation(
        x=0.95,
        y=0.05,
        text="Contact<br>No Power",
        xref="paper",
        yref="paper",
        showarrow=False,
    )
    fig.add_annotation(
        x=0.05,
        y=0.05,
        text="Below Average<br>Both",
        xref="paper",
        yref="paper",
        showarrow=False,
    )

    return fig


# =============================================================================
# 5. SITUATIONAL PERFORMANCE
# =============================================================================


def plot_count_specific_performance(pitcher_overall, batter_overall):
    """
    Compare performance in different count situations
    """
    # Pitcher performance in different counts
    count_cols_pitcher = [
        col for col in pitcher_overall.columns if "count" in col.lower()
    ]
    count_cols_batter = [
        col for col in batter_overall.columns if ("woba_" in col and "_" in col[-3:])
    ]

    if count_cols_batter:
        # Sample data for batters in key counts
        key_counts = ["woba_0_0", "woba_2_0", "woba_0_2", "woba_3_1"]
        available_counts = [col for col in key_counts if col in batter_overall.columns]

        if available_counts:
            count_data = batter_overall[["formatted_name"] + available_counts].head(20)
            count_melted = count_data.melt(
                id_vars=["formatted_name"], var_name="count", value_name="woba"
            )
            count_melted["count"] = (
                count_melted["count"].str.replace("woba_", "").str.replace("_", "-")
            )

            plt.figure(figsize=(12, 8))
            sns.boxplot(data=count_melted, x="count", y="woba")
            plt.title("Batter Performance by Count Situation")
            plt.xlabel("Count")
            plt.ylabel("wOBA")
            plt.xticks(rotation=45)

            return plt.gcf()

    return None


def plot_pressure_situation_analysis(pitcher_overall, batter_overall):
    """
    Performance in high-leverage situations
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Pitcher performance
    if (
        "pct_high_leverage" in pitcher_overall.columns
        and "overall_woba_against" in pitcher_overall.columns
    ):
        ax1.scatter(
            pitcher_overall["pct_high_leverage"],
            pitcher_overall["overall_woba_against"],
            alpha=0.6,
        )
        ax1.set_xlabel("High Leverage Situations %")
        ax1.set_ylabel("wOBA Against")
        ax1.set_title("Pitcher Performance Under Pressure")

        # Add trendline
        z = np.polyfit(
            pitcher_overall["pct_high_leverage"].fillna(0),
            pitcher_overall["overall_woba_against"].fillna(0),
            1,
        )
        p = np.poly1d(z)
        ax1.plot(
            pitcher_overall["pct_high_leverage"],
            p(pitcher_overall["pct_high_leverage"]),
            "r--",
            alpha=0.8,
        )

    # Batter performance
    if (
        "high_leverage_rate" in batter_overall.columns
        and "overall_woba" in batter_overall.columns
    ):
        ax2.scatter(
            batter_overall["high_leverage_rate"],
            batter_overall["overall_woba"],
            alpha=0.6,
            color="green",
        )
        ax2.set_xlabel("High Leverage Situations %")
        ax2.set_ylabel("wOBA")
        ax2.set_title("Batter Performance Under Pressure")

        # Add trendline
        z = np.polyfit(
            batter_overall["high_leverage_rate"].fillna(0),
            batter_overall["overall_woba"].fillna(0),
            1,
        )
        p = np.poly1d(z)
        ax2.plot(
            batter_overall["high_leverage_rate"],
            p(batter_overall["high_leverage_rate"]),
            "r--",
            alpha=0.8,
        )

    plt.tight_layout()
    return fig


# =============================================================================
# 6. MAIN EXECUTION FUNCTIONS
# =============================================================================


def create_all_visualizations(save_plots=True):
    """
    Generate all visualizations
    """
    # Load data
    data = load_data()

    if not data:
        return

    print("\n🎨 Creating visualizations...")

    plots = {}

    # 1. Pitcher Analysis
    print("📊 Pitcher visualizations...")
    try:
        plots["velocity_heatmap"] = plot_pitcher_velocity_heatmap(
            data["pitcher_by_pitch"]
        )
        plots["stuff_command"] = plot_stuff_vs_command_scatter(data["pitcher_overall"])

        # Individual pitcher radar (example)
        pitcher_names = data["pitcher_by_pitch"]["player_name"].unique()
        if len(pitcher_names) > 0:
            plots["arsenal_radar"] = plot_pitch_arsenal_radar(
                data["pitcher_by_pitch"], pitcher_names[0]
            )
    except Exception as e:
        print(f"Error in pitcher visualizations: {e}")

    # 2. Batter Analysis
    print("📊 Batter visualizations...")
    try:
        plots["contact_quality"] = plot_contact_quality_matrix(data["batter_by_pitch"])
        plots["discipline"] = plot_discipline_profiles(data["batter_overall"])
    except Exception as e:
        print(f"Error in batter visualizations: {e}")

    # 3. Matchup Analysis
    print("📊 Matchup visualizations...")
    try:
        plots["platoon"] = plot_platoon_advantage_heatmap(data["batter_vs_handedness"])
        plots["velocity_contact"] = plot_velocity_vs_contact(
            data["pitcher_overall"], data["batter_overall"]
        )
    except Exception as e:
        print(f"Error in matchup visualizations: {e}")

    # 4. League Analysis
    print("📊 League trend visualizations...")
    try:
        plots["pitch_mix"] = plot_pitch_mix_evolution(data["pitcher_by_pitch"])
        plots["power_contact"] = plot_power_vs_contact_spectrum(data["batter_overall"])
    except Exception as e:
        print(f"Error in league visualizations: {e}")

    # 5. Situational Analysis
    print("📊 Situational visualizations...")
    try:
        plots["count_performance"] = plot_count_specific_performance(
            data["pitcher_overall"], data["batter_overall"]
        )
        plots["pressure"] = plot_pressure_situation_analysis(
            data["pitcher_overall"], data["batter_overall"]
        )
    except Exception as e:
        print(f"Error in situational visualizations: {e}")

    if save_plots:
        save_all_plots(plots)

    print(
        f"\n✅ Created {len([p for p in plots.values() if p is not None])} visualizations!"
    )

    return plots


def save_all_plots(plots, output_dir="reports/figures"):
    """
    Save all plots to files
    """
    import os

    os.makedirs(output_dir, exist_ok=True)

    for name, plot in plots.items():
        if plot is not None:
            try:
                if hasattr(plot, "write_html"):  # Plotly figure
                    plot.write_html(f"{output_dir}/{name}.html")
                else:  # Matplotlib figure
                    plot.savefig(
                        f"{output_dir}/{name}.png", dpi=300, bbox_inches="tight"
                    )
                    plt.close(plot)
                print(f"💾 Saved {name}")
            except Exception as e:
                print(f"❌ Error saving {name}: {e}")


def show_data_summary(data):
    """
    Display summary of loaded data
    """
    print("\n" + "=" * 60)
    print("DATA SUMMARY")
    print("=" * 60)

    for name, df in data.items():
        print(f"\n📊 {name.upper()}")
        print(f"   Shape: {df.shape}")
        print(f"   Columns: {len(df.columns)}")
        if "formatted_name" in df.columns:
            print(f"   Players: {df['formatted_name'].nunique()}")
        elif "player_name" in df.columns:
            print(f"   Players: {df['player_name'].nunique()}")

    print("\n" + "=" * 60)


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    # Load data and show summary
    data = load_data()

    if data:
        show_data_summary(data)

        # Create all visualizations
        plots = create_all_visualizations(save_plots=True)

        # Show interactive plots (Plotly)
        interactive_plots = [
            "stuff_command",
            "contact_quality",
            "discipline",
            "pitch_mix",
            "power_contact",
        ]

        print("\n🎯 Interactive plots available:")
        for plot_name in interactive_plots:
            if plot_name in plots and plots[plot_name] is not None:
                print(f"   - plots['{plot_name}'].show()")

        print("\n📁 All plots saved to 'visualizations/' directory")
        print("   - HTML files for interactive plots")
        print("   - PNG files for static plots")
