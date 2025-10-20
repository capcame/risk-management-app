import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import warnings

warnings.filterwarnings('ignore')

# تنظیمات صفحه
st.set_page_config(
    page_title="مدیریت ریسک و سرمایه پیشرفته",
    page_icon="📊",
    layout="wide"
)

# استایل فارسی
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Vazirmatn:wght@400;700&display=swap');
* { 
    font-family: 'Vazirmatn', sans-serif !important;
}
.main {
    direction: rtl;
}
</style>
""", unsafe_allow_html=True)

def calculate_trades_to_target(initial_capital, target_percent, win_rate_percent, profit_percent, loss_percent, compound_type, commission_percent):
    win_rate = win_rate_percent / 100
    profit_rate = profit_percent / 100
    loss_rate = loss_percent / 100
    commission_rate = commission_percent / 100
    target_profit_total = initial_capital * (target_percent / 100)

    net_profit_rate = profit_rate - commission_rate
    net_loss_rate = loss_rate + commission_rate

    expected_return_per_trade = (win_rate * net_profit_rate) - ((1 - win_rate) * net_loss_rate)

    if expected_return_per_trade <= 0:
        return float('inf'), expected_return_per_trade

    if compound_type == "غیرمرکب":
        avg_profit_amount = initial_capital * expected_return_per_trade
        if avg_profit_amount <= 0:
            return float('inf'), expected_return_per_trade
        required_trades = target_profit_total / avg_profit_amount
    else:
        target_factor = 1 + (target_percent / 100)
        geometric_expected_return = (1 + net_profit_rate)**win_rate * (1 - net_loss_rate)**(1 - win_rate) - 1
        if geometric_expected_return <= 0:
             return float('inf'), expected_return_per_trade
        required_trades = np.log(target_factor) / np.log(1 + geometric_expected_return)

    return int(np.ceil(required_trades)), expected_return_per_trade

def simulate_trading(initial_capital, target_capital, win_rate, profit_percent, loss_percent, compound_type, commission_percent, max_trades, min_capital):
    capital = initial_capital
    capital_history = [capital]
    drawdown_history = [0]
    max_capital = initial_capital
    trades_count = 0

    profit_rate = profit_percent / 100
    loss_rate = loss_percent / 100
    commission_rate = commission_percent / 100

    for i in range(max_trades):
        if capital >= target_capital or capital <= min_capital:
            break

        trades_count += 1
        is_win = (i % 100) < (win_rate * 100)

        if is_win:
            if compound_type == "مرکب":
                capital *= (1 + profit_rate - commission_rate)
            else:
                capital += initial_capital * (profit_rate - commission_rate)
        else:
            if compound_type == "مرکب":
                capital *= (1 - loss_rate - commission_rate)
            else:
                capital -= initial_capital * (loss_rate + commission_rate)

        capital = max(capital, 0)
        capital_history.append(capital)

        max_capital = max(max_capital, capital)
        current_drawdown = (max_capital - capital) / max_capital * 100 if max_capital > 0 else 0
        drawdown_history.append(current_drawdown)

    max_drawdown = max(drawdown_history) if drawdown_history else 0
    return trades_count, capital, capital_history, drawdown_history, max_drawdown

def calculate_risk_metrics(initial_capital, target_profit_percent, profit_per_trade_percent,
                          loss_per_trade_percent, win_rate_percent, compound_type,
                          max_trades, trades_per_period, min_capital, commission_percent):
    
    required_trades_calc, expected_return = calculate_trades_to_target(
        initial_capital, target_profit_percent, win_rate_percent,
        profit_per_trade_percent, loss_per_trade_percent, compound_type, commission_percent
    )

    target_capital_value = initial_capital * (1 + target_profit_percent / 100)
    trades_count, final_capital, capital_history, drawdown_history, max_drawdown = simulate_trading(
        initial_capital, target_capital_value, win_rate_percent / 100,
        profit_per_trade_percent, loss_per_trade_percent, compound_type, commission_percent,
        max_trades, min_capital
    )

    profit_rate = profit_per_trade_percent / 100
    loss_rate = loss_per_trade_percent / 100
    effective_win_rate = (loss_rate / (profit_rate + loss_rate)) * 100 if (profit_rate + loss_rate) > 0 else 0

    risk_reward_ratio = profit_rate / loss_rate if loss_rate > 0 else float('inf')
    total_profit = final_capital - initial_capital
    total_profit_percent = (total_profit / initial_capital) * 100 if initial_capital > 0 else 0
    reached_target = final_capital >= target_capital_value

    return {
        'required_trades': trades_count,
        'calculated_trades': required_trades_calc,
        'reached_target': reached_target,
        'final_capital': final_capital,
        'target_capital': target_capital_value,
        'total_profit': total_profit,
        'total_profit_percent': total_profit_percent,
        'risk_reward_ratio': risk_reward_ratio,
        'effective_win_rate': effective_win_rate,
        'max_drawdown': max_drawdown,
        'expected_return_per_trade': expected_return * 100,
        'capital_history': capital_history,
        'drawdown_history': drawdown_history
    }

def generate_recommendations(results, win_rate_percent, effective_win_rate):
    recommendations = []
    
    if results['expected_return_per_trade'] <= 0:
        recommendations.append("❌ **هشدار:** استراتژی شما در بلندمدت زیان‌ده است!")
    
    if win_rate_percent < effective_win_rate:
        recommendations.append(f"⚠️ **توصیه:** برای سودآوری، وین‌ریت شما باید حداقل {effective_win_rate:.1f}% باشد")
    
    if results['max_drawdown'] > 20:
        recommendations.append("🔴 **ریسک بالا:** دراودان بیش از 20% - مدیریت ریسک خود را بازبینی کنید")
    elif results['max_drawdown'] > 10:
        recommendations.append("🟡 **ریسک متوسط:** دراودان قابل قبول ولی نیاز به نظارت دارد")
    else:
        recommendations.append("🟢 **عالی:** دراودان در محدوده ایمن قرار دارد")
    
    if results['risk_reward_ratio'] < 1:
        recommendations.append("📉 **توصیه:** نسبت Risk/Reward باید بیشتر از 1 باشد")
    
    return recommendations

def format_currency(amount):
    return f"{amount:,.0f} تومان"

def main():
    st.title("📊 سیستم مدیریت ریسک و سرمایه پیشرفته")
    st.markdown("---")
    
    with st.sidebar:
        st.header("⚙️ پارامترهای ورودی")
        
        initial_capital = st.number_input(
            "💰 سرمایه اولیه (تومان)",
            min_value=1000000,
            max_value=1000000000,
            value=100000000,
            step=1000000
        )
        
        target_profit_percent = st.slider(
            "🎯 درصد سود هدف",
            min_value=1,
            max_value=100,
            value=25,
            step=1
        )
        
        col1, col2 = st.columns(2)
        with col1:
            profit_per_trade_percent = st.slider(
                "📈 درصد سود هر معامله",
                min_value=0.1,
                max_value=20.0,
                value=3.0,
                step=0.1
            )
        with col2:
            loss_per_trade_percent = st.slider(
                "📉 درصد ضرر هر معامله",
                min_value=0.1,
                max_value=10.0,
                value=2.0,
                step=0.1
            )
        
        win_rate_percent = st.slider(
            "🎲 درصد وین‌ریت",
            min_value=1,
            max_value=99,
            value=60,
            step=1
        )
        
        compound_type = st.selectbox(
            "🔄 نوع محاسبه سود",
            ["مرکب", "غیرمرکب"]
        )
        
        commission_percent = st.slider(
            "💸 درصد کارمزد هر معامله",
            min_value=0.0,
            max_value=1.0,
            value=0.1,
                step=0.05
            )
        
        max_trades = st.number_input(
            "🔢 حداکثر تعداد معاملات شبیه‌سازی",
            min_value=10,
            max_value=10000,
            value=1000,
            step=10
        )
        
        min_capital = st.number_input(
            "🛑 حداقل سرمایه باقیمانده (تومان)",
            min_value=0,
            max_value=100000000,
            value=50000000,
            step=1000000
        )

    if st.button("🚀 محاسبه نتایج", use_container_width=True, type="primary"):
        with st.spinner("در حال محاسبه نتایج... لطفاً صبر کنید"):
            results = calculate_risk_metrics(
                initial_capital=initial_capital,
                target_profit_percent=target_profit_percent,
                profit_per_trade_percent=profit_per_trade_percent,
                loss_per_trade_percent=loss_per_trade_percent,
                win_rate_percent=win_rate_percent,
                compound_type=compound_type,
                max_trades=max_trades,
                trades_per_period=1,
                min_capital=min_capital,
                commission_percent=commission_percent
            )
            
            # نمایش نتایج
            st.success("✅ محاسبات با موفقیت انجام شد!")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "سرمایه نهایی",
                    format_currency(results['final_capital']),
                    f"{results['total_profit_percent']:.1f}%"
                )
                
            with col2:
                status = "✅ رسیده" if results['reached_target'] else "⏳ نرسیده"
                st.metric("وضعیت هدف", status)
                
            with col3:
                st.metric(
                    "تعداد معاملات انجام شده",
                    f"{results['required_trades']} معامله"
                )
            
            st.markdown("---")
            
            # نمودارها
            col1, col2 = st.columns(2)
            
            with col1:
                # نمودار سرمایه
                fig_capital = go.Figure()
                fig_capital.add_trace(go.Scatter(
                    y=results['capital_history'],
                    mode='lines',
                    name='سرمایه',
                    line=dict(color='#00ff88', width=3)
                ))
                fig_capital.add_hline(
                    y=results['target_capital'],
                    line_dash="dash",
                    line_color="red",
                    annotation_text="سرمایه هدف"
                )
                fig_capital.update_layout(
                    title="📈 روند سرمایه",
                    xaxis_title="تعداد معاملات",
                    yaxis_title="مقدار سرمایه (تومان)",
                    height=400
                )
                st.plotly_chart(fig_capital, use_container_width=True)
            
            with col2:
                # نمودار دراودان
                fig_drawdown = go.Figure()
                fig_drawdown.add_trace(go.Scatter(
                    y=results['drawdown_history'],
                    mode='lines',
                    name='دراودان',
                    line=dict(color='#ff4444', width=3),
                    fill='tozeroy'
                ))
                fig_drawdown.update_layout(
                    title="📉 روند دراودان",
                    xaxis_title="تعداد معاملات",
                    yaxis_title="درصد دراودان",
                    height=400
                )
                st.plotly_chart(fig_drawdown, use_container_width=True)
            
            # کارت‌های اطلاعاتی
            st.subheader("📊 تحلیل عملکرد")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.info(f"**امید ریاضی هر معامله:**\n{results['expected_return_per_trade']:.2f}%")
            
            with col2:
                st.info(f"**حداکثر دراودان:**\n{results['max_drawdown']:.1f}%")
            
            with col3:
                st.info(f"**نسبت Risk/Reward:**\n{results['risk_reward_ratio']:.2f}")
            
            with col4:
                st.info(f"**وین‌ریت مؤثر:**\n{results['effective_win_rate']:.1f}%")
            
            # توصیه‌ها
            st.subheader("💡 توصیه‌های مدیریت ریسک")
            recommendations = generate_recommendations(results, win_rate_percent, results['effective_win_rate'])
            
            for rec in recommendations:
                st.write(rec)

if __name__ == "__main__":
    main()
