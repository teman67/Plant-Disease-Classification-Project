import streamlit as st
from src.data_management import load_pkl_file


def load_test_evaluation(version):
    return load_pkl_file(f'jupyter_notebooks/outputs/{version}/evaluation.pkl')
