import streamlit as st


class MultiPage:
    """Framework for combining multiple Streamlit pages."""

    def __init__(self, app_name: str):
        self.app_name = app_name
        self.pages = []

        st.set_page_config(
            page_title=self.app_name,
            page_icon="🖥️",
            layout="wide"
        )

    def add_page(self, title: str, func):
        """Add a new page to the app."""
        self.pages.append({"title": title, "function": func})

    def run(self):
        """Run the selected page."""
        st.title(self.app_name)

        # Use sidebar menu
        selection = st.sidebar.radio(
            "Menu",
            self.pages,
            format_func=lambda page: page["title"]
        )

        # Run the selected page's function
        selection["function"]()
