"""
power_coupling: European day-ahead market-coupling research package.

The package name predates the July 2026 revision: the SMARD price series v1
believed to be German intraday indices are zonal day-ahead prices (DK1,
Belgium, DE/LU neighbours); see FINDINGS.md "Revision notes".

Research questions (post-revision framing):
  Q1  Renewable forecast errors and next-day auction prices (Granger, IRF, Bai-Perron)
  Q2  Cross-zonal day-ahead spreads under market coupling (cointegration, ECM, regime variance)
  Q3  "Shape spread" (cross-zonal hourly-vs-QH construction) and BESS dispatch LP
"""

__version__ = "0.1.0"
