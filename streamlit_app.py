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
        st.dataframe(raw_df.head(10).reset_index(), use_container_width=True)

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
    aggregation = st.radio("Aggregation method", ["mean", "sum", "last"], index=0, horizontal=True)

    interval_minutes = interval_options[selected_interval]
    aggregated_df = measurement_df.resample(f"{interval_minutes}min").agg(aggregation)
    aggregated_df = aggregated_df.dropna(how="all")

    if aggregated_df.empty:
        st.warning("The selected time range did not produce any 15-minute bins.")
        st.stop()

    st.subheader(f"{selected_interval} aggregated data")
    st.dataframe(aggregated_df.reset_index().rename(columns={aggregated_df.index.name: "Timestamp"}), use_container_width=True)

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
                if st.button(f"★ {parameter}", key=f"favorite_{parameter}", use_container_width=True):
                    updated = list(st.session_state.selected_parameters)
                    if parameter in updated:
                        updated.remove(parameter)
                    else:
                        updated.append(parameter)
                    st.session_state.selected_parameters = updated
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
        default=st.session_state.selected_parameters,
        key="selected_parameters",
        help="Start typing to search the dropdown, then select multiple parameters.",
    )

    if selected_parameters:
        chart_type = st.toggle("Use grouped block chart", value=False, help="Switch between a line chart and a grouped bar chart")
        if chart_type:
            chart_data = aggregated_df[selected_parameters].reset_index()
            chart_data = chart_data.melt(id_vars=[chart_data.columns[0]], var_name="Parameter", value_name="Value")
            chart = (
                alt.Chart(chart_data)
                .mark_bar()
                .encode(
                    x=alt.X("Parameter:N", title="Parameter"),
                    y=alt.Y("Value:Q", title="Value"),
                    color=alt.Color("Parameter:N", legend=None),
                    column=alt.Column(chart_data.columns[0], title=chart_data.columns[0]),
                )
                .properties(width=180, height=250)
                .resolve_scale(y="independent")
            )
            st.altair_chart(chart, use_container_width=True)
        else:
            st.line_chart(aggregated_df[selected_parameters])
    else:
        st.info("Select at least one parameter to display the trend plot.")
else:
    st.info("Upload a CSV file to begin.")
