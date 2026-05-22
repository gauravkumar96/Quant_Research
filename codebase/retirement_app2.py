# app.py
# Institutional-grade interview practical retirement planning app

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title='Institutional Retirement Planning', layout='wide')
np.random.seed(42)

# ---------------- Return Assumptions ----------------
ASSETS = {
    'US Equity': {'mu': 0.085, 'sigma': 0.16, 'fee': 0.001},
    'International Equity': {'mu': 0.075, 'sigma': 0.18, 'fee': 0.0015},
    'Investment Grade Bonds': {'mu': 0.045, 'sigma': 0.06, 'fee': 0.0008},
    'TIPS': {'mu': 0.04, 'sigma': 0.05, 'fee': 0.0007},
    'REITs': {'mu': 0.065, 'sigma': 0.15, 'fee': 0.0020},
    'Cash': {'mu': 0.03, 'sigma': 0.01, 'fee': 0.0},
}

# at the starting of the investment journey, the portfolio is more aggressive (higher equity) and gradually shifts to a more conservative allocation (higher bonds) as retirement approaches. The post-retirement allocation is more conservative to preserve capital while generating income.

PRE_RET_ALLOC = {
    'US Equity': 0.45,
    'International Equity': 0.15,
    'Investment Grade Bonds': 0.20,
    'TIPS': 0.05,
    'REITs': 0.10,
    'Cash': 0.05,
}

# by the end of the accumulation phase, the portfolio has shifted to a more conservative stance, and in retirement, it is even more focused on capital preservation and income generation, with a significant allocation to bonds and TIPS, while still maintaining some equity exposure for growth potential.

POST_RET_ALLOC = {
    'US Equity': 0.25,
    'International Equity': 0.10,
    'Investment Grade Bonds': 0.35,
    'TIPS': 0.15,
    'REITs': 0.10,
    'Cash': 0.05,
}

# ---------------- Utility Functions ----------------
def net_salary(gross, tax):
    return gross * (1 - tax)


def gross_withdrawal(net_need, withdrawal_tax):
    return net_need / (1 - withdrawal_tax)


def interpolate_alloc(year, total_years):
    if total_years <= 0:
        return POST_RET_ALLOC
    t = year / total_years
    alloc = {}
    for asset in PRE_RET_ALLOC:
        alloc[asset] = PRE_RET_ALLOC[asset] + t * (POST_RET_ALLOC[asset] - PRE_RET_ALLOC[asset])
    return alloc


def portfolio_return(alloc):
    return sum(alloc[a] * (ASSETS[a]['mu'] - ASSETS[a]['fee']) for a in alloc)


def portfolio_vol(alloc):
    return np.sqrt(sum((alloc[a] * ASSETS[a]['sigma']) ** 2 for a in alloc))


def accumulation_projection(age, retirement_age, salary, salary_growth, savings_rate, tax, current_assets):
    rows = []
    balance = current_assets
    years = retirement_age - age
    for y in range(years + 1):
        curr_age = age + y
        curr_salary = salary * ((1 + salary_growth) ** y)
        net = net_salary(curr_salary, tax)
        contribution = net * savings_rate
        alloc = interpolate_alloc(y, years)
        r = portfolio_return(alloc)
        rows.append([curr_age, round(curr_salary, 2), round(contribution, 2), round(balance, 2), r])
        balance = balance * (1 + r) + contribution
    return pd.DataFrame(rows, columns=['Age', 'Salary', 'Contribution', 'Portfolio', 'Return'])


def spending_curve(start_age, end_age, base_spend, inflation, withdrawal_tax):
    rows = []
    spend = base_spend
    for age in range(start_age, end_age + 1):
        gross_need = gross_withdrawal(spend, withdrawal_tax)
        rows.append([age, round(spend, 2), round(gross_need, 2)])
        spend *= (1 + inflation)
    return pd.DataFrame(rows, columns=['Age', 'Net Spending Need', 'Gross Withdrawal'])


def deterministic_corpus(spending_df, post_ret_return):
    pv = 0
    for i, cf in enumerate(spending_df['Gross Withdrawal'], start=1):
        pv += cf / ((1 + post_ret_return) ** i)
    return pv


def full_portfolio_curve(acc_df, spend_df, post_ret_return):
    # Unified accumulation + withdrawal trajectory with per-year cash-flow detail.
    # Each row's Portfolio is the start-of-year balance; Contribution and Withdrawal are
    # the cash flows during that year; Return Amount = Portfolio * Return %.
    rows = []

    # Accumulation: working years only — drop the retirement-age row from acc_df since the
    # retirement entry point is the first row of the withdrawal phase below.
    for i in range(len(acc_df) - 1):
        portfolio = float(acc_df['Portfolio'].iloc[i])
        contribution = float(acc_df['Contribution'].iloc[i])
        ret_pct = float(acc_df['Return'].iloc[i])
        rows.append({
            'Age': int(acc_df['Age'].iloc[i]),
            'Portfolio': round(portfolio, 2),
            'Contribution': round(contribution, 2),
            'Withdrawal': 0.0,
            'Return %': round(ret_pct * 100, 2),
            'Return Amount': round(portfolio * ret_pct, 2),
            'Phase': 'Accumulation',
        })

    # Withdrawal: deterministic, starting at retirement_age with corpus = projected assets.
    balance = float(acc_df.iloc[-1]['Portfolio'])
    for _, row in spend_df.iterrows():
        withdrawal = float(row['Gross Withdrawal'])
        rows.append({
            'Age': int(row['Age']),
            'Portfolio': round(balance, 2),
            'Contribution': 0.0,
            'Withdrawal': round(withdrawal, 2),
            'Return %': round(post_ret_return * 100, 2),
            'Return Amount': round(balance * post_ret_return, 2),
            'Phase': 'Withdrawal',
        })
        balance = balance * (1 + post_ret_return) - withdrawal
        if balance < 0:
            balance = 0

    return pd.DataFrame(rows)


def monte_carlo(current_assets, accumulation_df, spending_df, simulations):
    successes = 0
    endings = []
    paths = []

    # Hoist invariants out of the per-simulation loop
    n_acc = len(accumulation_df)
    contrib_arr = accumulation_df['Contribution'].values
    spend_arr = spending_df['Gross Withdrawal'].values
    n_spend = len(spend_arr)
    # Pre-compute glide-path allocations for each working year (skip last row = retirement entry)
    working_allocs = [interpolate_alloc(i, n_acc - 1) for i in range(n_acc - 1)]

    for _ in range(simulations):
        balance = current_assets

        # Accumulation: iterate working years only; the final row is the retirement entry point
        # (no contribution/growth applied there — matches the deterministic projection)
        for i, alloc in enumerate(working_allocs):
            port_ret = 0
            for asset, weight in alloc.items():
                params = ASSETS[asset]
                simulated = np.random.normal(params['mu'] - params['fee'], params['sigma'])
                port_ret += weight * simulated
            balance = balance * (1 + port_ret) + contrib_arr[i]

        horizon = n_spend

        survived = True
        yearly_balances = []
        for i in range(horizon):
            port_ret = 0
            for asset, weight in POST_RET_ALLOC.items():
                params = ASSETS[asset]
                simulated = np.random.normal(params['mu'] - params['fee'], params['sigma'])
                port_ret += weight * simulated

            withdrawal = spend_arr[min(i, n_spend - 1)]
            if withdrawal > balance:
                withdrawal = balance  # Can't withdraw more than the current balance
            balance = balance * (1 + port_ret) - withdrawal
            yearly_balances.append(balance)
            if balance <= 0:
                survived = False
                break

        if survived:
            successes += 1
        # Clamp depleted-portfolio endings to 0 so the distribution isn't polluted by large negatives
        endings.append(round(max(balance, 0), 2))
        if yearly_balances:
            paths.append(yearly_balances)

    return successes / simulations, endings, paths

# ---------------- Sidebar ----------------
st.sidebar.title('Client Inputs')
age = st.sidebar.number_input('Current Age', 35, 65, 50, step=1)
retirement_age = st.sidebar.number_input('Retirement Age', age + 1, 80, max(age + 1, 65), step=1)
life_expectancy = st.sidebar.number_input('Max Life Expectancy', retirement_age + 5, 110, max(retirement_age + 5, 95), step=1)
salary = st.sidebar.number_input('Current Salary ($)', 50000, 1000000, 150000, step=5000)
current_assets = st.sidebar.number_input('Current Investable Assets ($)', 0, 10000000, 200000, step=10000)
base_spending = st.sidebar.number_input('Desired Retirement Spending (today $)', 10000, 500000, 90000, step=5000)
salary_growth = st.sidebar.number_input('Salary Growth (%)', 0.0, 10.0, 3.5, step=0.1) / 100
savings_rate = st.sidebar.number_input('Savings Rate (% net income)', 0.0, 60.0, 25.0, step=0.5) / 100

st.sidebar.title('Tax Assumptions')
tax = st.sidebar.number_input('Effective Tax Rate (%)', 0.0, 60.0, 30.0, step=0.5, help='Blended federal + state + payroll rate applied to gross salary') / 100
withdrawal_tax = st.sidebar.number_input('Retirement Withdrawal Tax (%)', 0.0, 35.0, 20.0, step=0.5) / 100

st.sidebar.title('Retirement Assumptions')
inflation = st.sidebar.number_input('Inflation (%)', 0.0, 8.0, 3.0, step=0.1) / 100
simulations = st.sidebar.number_input('Monte Carlo Simulations', 1000, 10000, 3000, step=500)

# ---------------- Engine ----------------
acc_df = accumulation_projection(age, retirement_age, salary, salary_growth, savings_rate, tax, current_assets)
spend_df = spending_curve(retirement_age, life_expectancy, base_spending * ((1 + inflation) ** (retirement_age - age)), inflation, withdrawal_tax)
post_ret_return = portfolio_return(POST_RET_ALLOC)
required_corpus = deterministic_corpus(spend_df, post_ret_return)
projected_assets = acc_df.iloc[-1]['Portfolio']
funding_gap = required_corpus - projected_assets
success_prob, endings, mc_paths = monte_carlo(current_assets, acc_df, spend_df, simulations)

# Build the full portfolio curve: accumulation phase + deterministic withdrawal phase
full_curve_df = full_portfolio_curve(acc_df, spend_df, post_ret_return)

# ---------------- Dashboard ----------------
st.title('Retirement Planning')

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric('Required Retirement Corpus', f'${required_corpus:,.0f}')
c2.metric('Projected Assets at Retirement', f'${projected_assets:,.0f}')
c3.metric('Funding Gap', f'${max(funding_gap,0):,.0f}')
c4.metric('Success Probability', f'{success_prob:.1%}')
c5.metric('Effective Tax Rate', f'{tax:.1%}')

def csv_download(df, label, filename, key):
    st.download_button(
        label=f'Download {label} (CSV)',
        data=df.to_csv(index=False).encode('utf-8'),
        file_name=filename,
        mime='text/csv',
        key=key,
    )

fig_full = px.line(
    full_curve_df, x='Age', y='Portfolio', color='Phase',
    title='Portfolio Curve'
)

fig_full.add_vline(x=retirement_age, line_dash='dash', line_color='gray', annotation_text='Retirement')
st.plotly_chart(fig_full, use_container_width=True)
st.dataframe(full_curve_df, use_container_width=True, hide_index=True)
csv_download(full_curve_df, 'Complete Portfolio Curve', 'portfolio_curve.csv', 'dl_full')

fig1 = px.area(acc_df, x='Age', y='Portfolio', title='Accumulation Path')
st.plotly_chart(fig1, use_container_width=True)
csv_download(acc_df, 'Accumulation Path', 'accumulation_path.csv', 'dl_acc')

fig2 = px.line(spend_df, x='Age', y='Gross Withdrawal', title='Retirement Withdrawal Curve')
st.plotly_chart(fig2, use_container_width=True)
csv_download(spend_df, 'Withdrawal Curve', 'withdrawal_curve.csv', 'dl_spend')

mc_df = pd.DataFrame({'Ending Wealth': endings})
fig3 = px.histogram(mc_df, x='Ending Wealth', title='Monte Carlo Ending Wealth Distribution')
st.plotly_chart(fig3, use_container_width=True)
csv_download(mc_df, 'Ending Wealth Distribution', 'monte_carlo_endings.csv', 'dl_mc')

# Glide Path
alloc_rows = []
for y in range(retirement_age - age + 1):
    alloc = interpolate_alloc(y, retirement_age - age)
    row = {'Age': age + y}
    row.update(alloc)
    alloc_rows.append(row)
alloc_df = pd.DataFrame(alloc_rows)
fig4 = px.area(alloc_df, x='Age', y=list(PRE_RET_ALLOC.keys()), title='Glide Path Allocation')
st.plotly_chart(fig4, use_container_width=True)
csv_download(alloc_df, 'Glide Path Allocation', 'glide_path.csv', 'dl_glide')

# Recommendation
st.subheader('Advisory Recommendation')
if success_prob > 0.85:
    st.success('Retirement plan appears robust under modeled assumptions.')
elif success_prob > 0.65:
    st.warning('Plan is moderately viable. Consider modest contribution increases or delayed retirement.')
else:
    st.error('Plan appears weak. Recommended interventions: increase savings, reduce spending, or retire later.')



with st.expander('Assumptions & Limitations'):
    st.write('''
    Assumptions:
    - Forward-looking capital market assumptions
    - Simplified zero-correlation asset aggregation
    - Tax approximations using effective rates
    - Retirement spending grows at inflation only
    - Deterministic retirement horizon (retirement age to max life expectancy)

    Limitations:
    - No detailed tax lot/account optimization
    - No annuity optimization
    - No regime-switching return models
    - No explicit correlation matrix between asset classes
    ''')
