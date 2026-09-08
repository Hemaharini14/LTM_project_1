"""
Generates exploratory data analysis plots from the cleaned datasets and
saves them as PNGs in outputs/plots/. Run after clean_airlines.py and
clean_hotels.py (or just run main.py, which sequences everything).
"""
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless rendering, no display needed
import matplotlib.pyplot as plt
import seaborn as sns

from config import AIRLINES_CLEAN_PATH, HOTELS_CLEAN_PATH, PLOTS_DIR

sns.set_theme(style="whitegrid")


def plot_delay_by_airline(df: pd.DataFrame):
    rates = df.groupby("Airline")["Delay"].mean().sort_values(ascending=False)
    plt.figure(figsize=(10, 5))
    sns.barplot(x=rates.index, y=rates.values, hue=rates.index, palette="viridis", legend=False)
    plt.title("Historical Delay Rate by Airline")
    plt.ylabel("Delay rate")
    plt.xlabel("Airline code")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/delay_rate_by_airline.png", dpi=120)
    plt.close()


def plot_delay_by_time_and_day(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    time_order = ["Morning", "Afternoon", "Evening", "Night"]
    time_rates = df.groupby("time_of_day")["Delay"].mean().reindex(time_order)
    sns.barplot(x=time_rates.index, y=time_rates.values, ax=axes[0], hue=time_rates.index,
                palette="mako", legend=False)
    axes[0].set_title("Delay Rate by Time of Day")
    axes[0].set_ylabel("Delay rate")

    day_names = {1: "Mon", 2: "Tue", 3: "Wed", 4: "Thu", 5: "Fri", 6: "Sat", 7: "Sun"}
    day_rates = df.groupby("DayOfWeek")["Delay"].mean()
    day_rates.index = day_rates.index.map(day_names)
    sns.barplot(x=day_rates.index, y=day_rates.values, ax=axes[1], hue=day_rates.index,
                palette="mako", legend=False)
    axes[1].set_title("Delay Rate by Day of Week")
    axes[1].set_ylabel("Delay rate")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/delay_by_time_and_day.png", dpi=120)
    plt.close()


def plot_top_routes(df: pd.DataFrame, top_n: int = 15):
    top_routes = df["route"].value_counts().head(top_n)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_routes.values, y=top_routes.index, hue=top_routes.index,
                palette="crest", legend=False)
    plt.title(f"Top {top_n} Busiest Routes in Dataset")
    plt.xlabel("Number of flights")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/top_routes.png", dpi=120)
    plt.close()


def plot_hotel_adr_distribution(df: pd.DataFrame):
    plt.figure(figsize=(9, 5))
    sns.histplot(data=df, x="adr", hue="hotel", bins=60, kde=True, element="step")
    plt.xlim(0, 400)
    plt.title("Nightly Rate (ADR) Distribution by Hotel Type")
    plt.xlabel("Average Daily Rate (USD)")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/hotel_adr_distribution.png", dpi=120)
    plt.close()


def plot_hotel_cancellation_by_segment(df: pd.DataFrame):
    rates = df.groupby("market_segment")["is_canceled"].mean().sort_values(ascending=False)
    plt.figure(figsize=(9, 5))
    sns.barplot(x=rates.values, y=rates.index, hue=rates.index, palette="rocket", legend=False)
    plt.title("Cancellation Rate by Market Segment")
    plt.xlabel("Cancellation rate")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/hotel_cancellation_by_segment.png", dpi=120)
    plt.close()


def run():
    print("=" * 60)
    print("GENERATING EDA VISUALS")
    print("=" * 60)
    airlines = pd.read_csv(AIRLINES_CLEAN_PATH)
    hotels = pd.read_csv(HOTELS_CLEAN_PATH)

    plot_delay_by_airline(airlines)
    plot_delay_by_time_and_day(airlines)
    plot_top_routes(airlines)
    plot_hotel_adr_distribution(hotels)
    plot_hotel_cancellation_by_segment(hotels)

    print(f"Saved 5 plots -> {PLOTS_DIR}")


if __name__ == "__main__":
    run()