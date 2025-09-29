"""
Core modules for the Portfolio Management Suite
"""

from .evm_engine import (
    # Main APIs
    perform_batch_calculation,
    perform_complete_evm_analysis,
    calculate_evm_metrics,

    # EVM calculation functions
    calculate_pv_linear,
    calculate_pv_scurve,
    find_earned_schedule_linear,
    find_earned_schedule_scurve,
    calculate_earned_schedule_metrics,

    # Mathematical functions
    scurve_cdf,
    calculate_durations,
    calculate_present_value,

    # Utility functions
    safe_divide,
    safe_financial_metrics,
    validate_numeric_input,
    is_valid_finite_number,
    parse_date_any,
    add_months_approx,
    safe_calculate_forecast_duration,
)

__all__ = [
    # Main APIs
    'perform_batch_calculation',
    'perform_complete_evm_analysis',
    'calculate_evm_metrics',

    # EVM calculation functions
    'calculate_pv_linear',
    'calculate_pv_scurve',
    'find_earned_schedule_linear',
    'find_earned_schedule_scurve',
    'calculate_earned_schedule_metrics',

    # Mathematical functions
    'scurve_cdf',
    'calculate_durations',
    'calculate_present_value',

    # Utility functions
    'safe_divide',
    'safe_financial_metrics',
    'validate_numeric_input',
    'is_valid_finite_number',
    'parse_date_any',
    'add_months_approx',
    'safe_calculate_forecast_duration',
]