import streamlit as st
import pandas as pd
import altair as alt

st.set_page_config(page_title="PQM Readings", page_icon=":chart_with_upwards_trend:")


def normalize_name(name: str) -> str:
    return "".join(ch.lower() for ch in str(name) if ch.isalnum())


def find_column(columns, candidates):
    normalized_columns = {normalize_name(col): col for col in columns}
    for candidate in candidates:
        normalized_candidate = normalize_name(candidate)
        if normalized_candidate in normalized_columns:
            return normalized_columns[normalized_candidate]
    for candidate in candidates:
        lower_candidate = candidate.lower()
        for col in columns:
            col_lower = str(col).lower()
            if lower_candidate in col_lower or col_lower in lower_candidate:
                return col
    return None


def prepare_pqm_data(uploaded_files):
    dataframes = []
    for uploaded_file in uploaded_files:
        file_df = pd.read_csv(uploaded_file, index_col=False, skipinitialspace=True)
        if file_df.empty:
            raise ValueError(f"The uploaded file '{uploaded_file.name}' is empty.")
        dataframes.append(file_df)

    df = pd.concat(dataframes, ignore_index=True)

    if df.empty:
        raise ValueError("The uploaded files are empty.")

    time_column = find_column(
        df.columns,
        ["Timestamp", "Datetime", "DateTime", "Date", "Time"],
    )
    if time_column is None:
        available = ", ".join(str(column) for column in df.columns[:20])
        raise ValueError(f"Could not find a timestamp column. Available: {available}")

    try:
        df[time_column] = pd.to_datetime(df[time_column], errors="coerce")
    except Exception as exc:
        raise ValueError(f"Could not parse the time column: {exc}") from exc

    valid_count = df[time_column].notna().sum()
    total_count = len(df)
    invalid_count = total_count - valid_count

    df = df.dropna(subset=[time_column]).copy()
    if df.empty:
        raise ValueError(f"No valid timestamps were found. All {total_count} rows had invalid timestamps.")

    if invalid_count > 0:
        st.warning(f"⚠️ {invalid_count} out of {total_count} rows had invalid timestamps and were removed.")

    df = df.sort_values(time_column)
    df = df.set_index(time_column)

    measurement_columns = [column for column in df.columns if column != time_column]
    numeric_df = df[measurement_columns].apply(pd.to_numeric, errors="coerce")
    numeric_df = numeric_df.dropna(how="all", axis=1)

    if numeric_df.empty:
        raise ValueError("No numeric measurement values were found in the uploaded file.")

    return df, numeric_df, time_column, list(numeric_df.columns)


def compute_series_delta(series: pd.Series):
    values = series.dropna()
    if len(values) < 2:
        return None
    return values.iloc[-1] - values.iloc[0]


def compute_pt_sum(series: pd.Series):
    values = series.dropna()
    if values.empty:
        return None
    return values.sum()


def format_timedelta(duration: pd.Timedelta) -> str:
    seconds = int(duration.total_seconds())
    if seconds % 3600 == 0:
        return f"{seconds // 3600} hours"
    if seconds % 60 == 0:
        return f"{seconds // 60} minutes"
    return f"{seconds} seconds"


def compute_pt_peaks(series: pd.Series, top_n: int = 3):
    values = series.dropna()
    peaks = []
    if values.empty:
        return peaks

    if len(values) > 1:
        median_delta = values.index.to_series().diff().median()
        if pd.isna(median_delta):
            median_delta = pd.Timedelta(minutes=1)
    else:
        median_delta = pd.Timedelta(minutes=1)

    visited = set()
    for timestamp in values.sort_values(ascending=False).index:
        if timestamp in visited:
            continue
        peak_value = values.loc[timestamp]
        position = values.index.get_loc(timestamp)
        start = position
        while start > 0 and values.iloc[start - 1] == peak_value:
            start -= 1
        end = position
        while end + 1 < len(values) and values.iloc[end + 1] == peak_value:
            end += 1

        peak_range = values.index[start : end + 1]
        duration = median_delta * len(peak_range)
        peak_values = values.iloc[start : end + 1]
        peaks.append({
            "value": float(peak_value),
            "start": peak_range[0],
            "end": peak_range[-1],
            "duration": duration,
            "avg": float(peak_values.mean()),
        })

        for index in peak_range:
            visited.add(index)

        if len(peaks) >= top_n:
            break

    return peaks


def compute_pt_average_peak(series: pd.Series, bin_width: float = 0.5):
    values = series.dropna()
    if values.empty:
        return None

    business_values = values[values.index.weekday < 5]
    business_values = business_values[(business_values.index.hour >= 9) & (business_values.index.hour < 20)]
    if business_values.empty:
        return None

    min_val = business_values.min()
    max_val = business_values.max()
    if min_val == max_val:
        return float(min_val), (min_val, max_val)

    bins = pd.interval_range(start=min_val, end=max_val + bin_width, freq=bin_width, closed="left")
    labels = pd.cut(business_values, bins, right=False)
    counts = labels.value_counts().sort_values(ascending=False)
    if counts.empty:
        return None

    most_frequent_bin = counts.index[0]
    avg_peak = business_values[labels == most_frequent_bin].mean()
    return float(avg_peak), most_frequent_bin


def toggle_selected_parameter(parameter: str):
    selected = st.session_state.selected_parameters
    if parameter in selected:
        selected.remove(parameter)
    else:
        selected.append(parameter)
    st.session_state.selected_parameters = selected


st.title("PQM Readings Dashboard")
st.write("Upload CSV files with 1-minute PQM readings to combine, aggregate into 15-minute intervals, and plot.")

folder_files = st.file_uploader(
    "Folder upload",
    type=["csv"],
    accept_multiple_files="directory",
)
single_file = st.file_uploader("Individual upload", type=["csv"])
uploaded_files = folder_files + ([single_file] if single_file is not None else [])

if uploaded_files:
    try:
        raw_df, measurement_df, time_column, target_columns = prepare_pqm_data(uploaded_files)
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

    st.success(f"{len(uploaded_files)} file(s) loaded successfully.")
    
    time_index = measurement_df.index
    time_range = f"{time_index[0]} to {time_index[-1]}"
    time_delta_samples = ((time_index[1:10] - time_index[0:9]).total_seconds() / 60).values
    
    with st.expander("📊 Data diagnostics"):
        st.write(f"**Files loaded:** {len(uploaded_files)}")
        st.write(f"**Timestamp column:** {time_column}")
        st.write(f"**Time range:** {time_range}")
        st.write(f"**Total rows:** {len(measurement_df)}")
        st.write(f"**Columns extracted:** {', '.join(target_columns)}")
        st.write(f"**Time intervals (min):** {time_delta_samples.mean():.1f} (expected: 1.0)")
        st.dataframe(raw_df.head(10).reset_index(), width="stretch")

    time_index = measurement_df.index
    if len(time_index) < 2:
        st.warning("The uploaded data contains too few rows to build a meaningful 15-minute trend.")
        st.stop()

    interval_options = {
        "1 minute": 1,
        "5 minutes": 5,
        "10 minutes": 10,
        "15 minutes": 15,
        "30 minutes": 30,
        "1 hour": 60,
        "2 hours": 120,
        "4 hours": 240,
        "6 hours": 360,
        "12 hours": 720,
        "24 hours": 1440,
    }
    selected_interval = st.selectbox(
        "Time interval",
        options=list(interval_options),
        index=3,
    )
    aggregation_options = {
        "mean": "mean",
        "sum": "sum",
        "last": "last",
        "highest value": "max",
    }
    aggregation_label = st.radio(
        "Aggregation method",
        options=list(aggregation_options),
        index=0,
        horizontal=True,
    )
    aggregation_method = aggregation_options[aggregation_label]

    interval_minutes = interval_options[selected_interval]
    aggregated_df = measurement_df.resample(f"{interval_minutes}min").agg(aggregation_method)
    aggregated_df = aggregated_df.dropna(how="all")

    if aggregated_df.empty:
        st.warning("The selected time range did not produce any 15-minute bins.")
        st.stop()

    st.subheader(f"{selected_interval} aggregated data")
    st.dataframe(aggregated_df.reset_index().rename(columns={aggregated_df.index.name: "Timestamp"}), width="stretch")

    st.subheader("Trend plot")

    if "favorite_parameters" not in st.session_state:
        st.session_state.favorite_parameters = []
    if "selected_parameters" not in st.session_state:
        st.session_state.selected_parameters = []

    all_parameters = list(aggregated_df.columns)
    searched_text = st.text_input("Search parameters", placeholder="Type to filter the parameter list")
    filtered_parameters = [
        parameter
        for parameter in all_parameters
        if not searched_text or searched_text.lower() in parameter.lower()
    ]

    if not filtered_parameters:
        st.warning("No parameters match the search filter.")

    selected_parameters = st.session_state.get("selected_parameters", [])
    selected_parameters = [parameter for parameter in selected_parameters if parameter in filtered_parameters]

    st.caption("Search and select multiple parameters from a scrollable dropdown.")

    if st.session_state.favorite_parameters:
        st.write("Favorite parameters")
        favorite_cols = st.columns(min(4, max(1, len(st.session_state.favorite_parameters))))
        for index, parameter in enumerate(st.session_state.favorite_parameters):
            with favorite_cols[index % len(favorite_cols)]:
                if st.button(f"★ {parameter}", key=f"favorite_{parameter}", width="stretch", on_click=toggle_selected_parameter, args=(parameter,)):
                    pass
    else:
        st.info("Favorite parameters will appear here once you add them.")

    if st.button("Add selected parameters to favorites"):
        for parameter in selected_parameters:
            if parameter not in st.session_state.favorite_parameters:
                st.session_state.favorite_parameters.append(parameter)
        st.success("Favorites updated.")

    selected_parameters = st.multiselect(
        "Parameters to display",
        options=filtered_parameters,
        default=[param for param in st.session_state.selected_parameters if param in filtered_parameters],
        key="selected_parameters",
        help="Start typing to search the dropdown, then select multiple parameters.",
    )

    if selected_parameters:
        chart_type = st.toggle("Use grouped block chart", value=False, help="Switch between a line chart and a grouped bar chart")
        chart_data = aggregated_df[selected_parameters].reset_index()
        time_column_name = chart_data.columns[0]

        if chart_type:
            chart_data = chart_data.melt(id_vars=[time_column_name], var_name="Parameter", value_name="Value")
            chart = (
                alt.Chart(chart_data)
                .mark_bar()
                .encode(
                    x=alt.X("Parameter:N", title="Parameter"),
                    y=alt.Y("Value:Q", title="Value"),
                    color=alt.Color("Parameter:N", legend=None),
                    column=alt.Column(time_column_name, title=time_column_name),
                )
                .properties(width=180, height=250)
                .resolve_scale(y="independent")
            )
        else:
            chart = (
                alt.Chart(chart_data)
                .mark_line(point=True)
                .encode(
                    x=alt.X(f"{time_column_name}:T", title=time_column_name),
                    y=alt.Y("value:Q", title="Value"),
                    color=alt.Color("variable:N", title="Parameter"),
                    tooltip=[alt.Tooltip(f"{time_column_name}:T", title=time_column_name), alt.Tooltip("variable:N", title="Parameter"), alt.Tooltip("value:Q", title="Value")],
                )
                .transform_fold(selected_parameters, as_=["variable", "value"])
                .interactive()
            )

        st.altair_chart(chart, width="stretch")

        st.markdown("---")
        st.write("### Total usage summary")
        summary_shown = False
        for parameter in selected_parameters:
            series = aggregated_df[parameter].dropna()
            normalized = normalize_name(parameter)
            if normalized == normalize_name("Import kWh"):
                if len(series) >= 2:
                    delta = compute_series_delta(series)
                    st.write(f"**{parameter}:** {delta:.3f} (last - first from {series.index[0]} to {series.index[-1]})")
                else:
                    st.write(f"**{parameter}:** Not enough data to calculate total usage.")
                summary_shown = True
            elif normalized == normalize_name("PT (kW)"):
                total = compute_pt_sum(series)
                if total is not None:
                    st.write(f"**{parameter}:** {total:.3f} (sum of all PT entries from {series.index[0]} to {series.index[-1]})")
                    peaks = compute_pt_peaks(series, top_n=3)
                    if peaks:
                        st.write("#### Top 3 PT peaks")
                        for idx, peak in enumerate(peaks, start=1):
                            st.write(
                                f"{idx}. Peak {peak['value']:.3f} kW — "
                                f"avg {peak['avg']:.3f} kW over {format_timedelta(peak['duration'])} "
                                f"({peak['start']} to {peak['end']})"
                            )
                        avg_peak_result = compute_pt_average_peak(series, bin_width=0.5)
                        if avg_peak_result is not None:
                            avg_peak_value, peak_bin = avg_peak_result
                            st.write(f"#### Avg Peak = {avg_peak_value:.3f} kW (most frequent range during Office Hour (9am- 8pm) {peak_bin})")
                else:
                    st.write(f"**{parameter}:** Not enough data to calculate total usage.")
                summary_shown = True
        if not summary_shown:
            st.write("No selected Import kWh or PT (kW) parameters for a total usage summary.")
    else:
        st.info("Select at least one parameter to display the trend plot.")
else:
    st.info("Upload a CSV file to begin.")
