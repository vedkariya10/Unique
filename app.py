import streamlit as st

st.set_page_config(
    page_title="SwiftSole Intelligence Dashboard",
    page_icon="👟",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar navigation ────────────────────────────────────────────────────────
st.sidebar.image(
    "https://via.placeholder.com/260x60/111111/E8FF00?text=SWIFTSOLE",
    use_column_width=True,
)
st.sidebar.markdown("---")
st.sidebar.markdown("### Analytics Dashboard")

PAGES = {
    "📊 Descriptive Analysis":   "page_descriptive",
    "🔍 Diagnostic Analysis":    "page_diagnostic",
    "🤖 Predictive Modelling":   "page_predictive",
    "🎯 Prescriptive Actions":   "page_prescriptive",
    "📥 New Data Upload & Score":"page_upload",
}

selection = st.sidebar.radio("Navigate to", list(PAGES.keys()))
st.sidebar.markdown("---")
st.sidebar.caption("SwiftSole · Data Intelligence v2.0")
st.sidebar.caption("Dataset: 2,000 Indian respondents · Seed 42")

# ── Route to page ─────────────────────────────────────────────────────────────
page_key = PAGES[selection]

if page_key == "page_descriptive":
    from pages.page_descriptive import render
    render()
elif page_key == "page_diagnostic":
    from pages.page_diagnostic import render
    render()
elif page_key == "page_predictive":
    from pages.page_predictive import render
    render()
elif page_key == "page_prescriptive":
    from pages.page_prescriptive import render
    render()
elif page_key == "page_upload":
    from pages.page_upload import render
    render()
