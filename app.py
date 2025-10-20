```python
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
    font-family: 'Vazirmatn', sans-serif;
}

.main-header {
    text-align: center;
    color: #1f77b4;
    margin-bottom: 2rem;
}

.sidebar-header {
    color: #2e86ab;
    margin-bottom: 1rem;
}

.metric-card {
    background-color: #f8f9fa;
    padding: 1rem;
    border-radius: 10px;
    border-left: 4px solid #1f77b4;
    margin-bottom: 1rem;
}

.warning-box {
    background-color: #fff3cd;
    border: 1px solid #ffeaa7;
    border-radius: 5px;
    padding: 1rem;
    margin: 1rem 0;
}

.success-box {
    background-color: #d1ecf1;
    border: 1px solid #bee5eb;
    border-radius: 5px;
    padding: 1rem;
    margin: 1rem 0;
}

.recommendation-box {
    background-color: #e8f5e8;
    border: 1px solid #4caf50;
    border-radius: 10px;
    padding: 1.5rem;
    margin: 1rem 0;
}

.analysis-box {
    background-color: #fff3e0;
    border: 1px solid #ff9800;
    border-radius: 10px;
    padding: 1.5rem;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

def calculate_exact_trades(initial_capital, target_percent, win_rate_percent, profit_percent, loss_percent, compound_type):
    """
    محاسبات قطعی و دقیق تعداد معاملات - نسخه اصلاح شده
    """
    win_rate = win_rate_percent / 100
    target_capital = initial_capital * (1 + target_percent/100)
    
    # سود مورد انتظار هر معامله
    expected_return_per_trade = (win_rate * (profit_percent/100)) - ((1 - win_rate) * (loss_percent/100))
    
    if expected_return_per_trade <= 0:
        return 1000, expected_return_per_trade  # عدد بزرگ نشان دهنده عدم امکان رسیدن به هدف
    
    if compound_type == "غیرمرکب":
        # غیرمرکب - محاسبه ساده
        required_trades = (target_percent/100) / expected_return_per_trade
    else:
        # مرکب - فرمول دقیق
        required_trades = np.log(target_capital / initial_capital) / np.log(1 + expected_return_per_trade)
    
    return int(np.ceil(required_trades)), expected_return_per_trade

def simulate_trading(initial_capital, target_capital, win_rate, profit_percent, loss_percent, compound_type, max_trades=1000):
    """
    شبیه‌سازی معاملات - نسخه کاملاً اصلاح شده و پایدار
    """
    capital = initial_capital
    capital_history = [capital]
    drawdown_history = [0]
    
    max_capital = initial_capital
    max_drawdown = 0
    trades_count = 0
    
    # برای غیرمرکب - محاسبه سود خالص
    net_profit = 0
    
    # الگوی قطعی برای نتایج یکسان
    # ایجاد دنباله‌ای از معاملات بر اساس وین‌ریت
    trade_results = []
    for i in range(max_trades):
        if (i % 100) < (win_rate * 100):
            trade_results.append(1)  # سود
        else:
            trade_results.append(-1)  # ضرر
    
    for i in range(max_trades):
        if capital >= target_capital:
            break
            
        trades_count += 1
        
        # شبیه‌سازی معامله با الگوی قطعی
        if trade_results[i] == 1:
            # معامله سودده
            if compound_type == "مرکب":
                capital *= (1 + profit_percent/100)
            else:
                net_profit += initial_capital * (profit_percent/100)
                capital = initial_capital + net_profit
        else:
            # معامله ضررده
            if compound_type == "مرکب":
                capital *= (1 - loss_percent/100)
            else:
                net_profit -= initial_capital * (loss_percent/100)
                capital = initial_capital + net_profit
        
        # جلوگیری از سرمایه منفی
        capital = max(capital, 0)
        
        # محاسبه دراودان
        if capital > max_capital:
            max_capital = capital
        
        current_drawdown = (max_capital - capital) / max_capital * 100 if max_capital > 0 else 0
        if current_drawdown > max_drawdown:
            max_drawdown = current_drawdown
        
        capital_history.append(capital)
        drawdown_history.append(current_drawdown)
        
        # اگر سرمایه به صفر رسید، متوقف شو
        if capital <= 0:
            break
    
    return trades_count, capital, capital_history, drawdown_history, max_drawdown

def calculate_risk_metrics(initial_capital, target_profit_percent, profit_per_trade_percent,
                          loss_per_trade_percent, win_rate_percent, compound_type,
                          max_trades=None, trades_per_period=None, min_capital=None,
                          commission_percent=0):
    """
    محاسبات اصلی مدیریت ریسک و سرمایه - نسخه کاملاً اصلاح شده
    """
    # تبدیل درصدها
    target_profit = target_profit_percent / 100
    profit_per_trade = profit_per_trade_percent / 100
    loss_per_trade = loss_per_trade_percent / 100
    win_rate = win_rate_percent / 100
    
    # محاسبه قطعی تعداد معاملات
    exact_trades, expected_return = calculate_exact_trades(
        initial_capital, target_profit_percent, win_rate_percent, 
        profit_per_trade_percent, loss_per_trade_percent, compound_type
    )
    
    # محاسبات پایه
    risk_reward_ratio = profit_per_trade / loss_per_trade if loss_per_trade > 0 else float('inf')
    target_capital_value = initial_capital * (1 + target_profit)
    
    # شبیه‌سازی معاملات
    trades_count, final_capital, capital_history, drawdown_history, max_drawdown = simulate_trading(
        initial_capital, target_capital_value, win_rate, 
        profit_per_trade_percent, loss_per_trade_percent, compound_type,
        max_trades=max_trades or 1000
    )
    
    # استفاده از تعداد معاملات تئوریک به جای تعداد معاملات شبیه‌سازی شده
    required_trades = exact_trades if expected_return > 0 else trades_count
    
    # محاسبه وین‌ریت مؤثر (حداقل وین‌ریت برای سودآوری)
    if profit_per_trade > 0 and loss_per_trade > 0:
        effective_win_rate = (loss_per_trade / (profit_per_trade + loss_per_trade)) * 100
    else:
        effective_win_rate = 0
    
    # نتایج نهایی
    total_profit = final_capital - initial_capital
    total_profit_percent = (total_profit / initial_capital) * 100 if initial_capital > 0 else 0
    reached_target = final_capital >= target_capital_value
    
    # محاسبه احتمال موفقیت
    win_rate_diff = win_rate_percent - effective_win_rate
    if win_rate_diff > 0:
        success_probability = min(95, 50 + (win_rate_diff * 1.5))
    else:
        success_probability = max(5, 50 + (win_rate_diff * 2))
    
    # وضعیت پایداری
    if success_probability > 75 and max_drawdown < 10:
        stability_status = "پایدار 🟢"
    elif success_probability > 60 and max_drawdown < 20:
        stability_status = "نیمه‌پایدار 🟡"
    else:
        stability_status = "ناپایدار 🔴"
    
    # زمان تقریبی
    estimated_time = ""
    if trades_per_period and trades_per_period > 0:
        periods_needed = required_trades / trades_per_period
        if periods_needed < 1:
            estimated_time = "کمتر از 1 دوره"
        else:
            estimated_time = f"{periods_needed:.1f} دوره"
    
    # هشدارها
    warnings_list = []
    
    # بررسی سودآوری استراتژی
    if expected_return <= 0:
        warnings_list.append("⛔ استراتژی شما سودآور نیست! سود مورد انتظار منفی است")
    
    if win_rate_percent < effective_win_rate:
        warnings_list.append(f"⚠️ وین‌ریت شما برای سودآوری کافی نیست. حداقل وین‌ریت مورد نیاز: {effective_win_rate:.1f}%")
    
    if not reached_target and trades_count >= (max_trades or 1000):
        warnings_list.append("⚠️ با تعداد معاملات مجاز به هدف نرسیدید")
    
    if max_drawdown > 25:
        warnings_list.append("⚠️ افت سرمایه بیش از حد مجاز است")
    
    if success_probability < 40:
        warnings_list.append("⚠️ احتمال موفقیت پایین است")
    
    if min_capital and final_capital < min_capital:
        warnings_list.append("⛔ سرمایه به حداقل مجاز رسید")
    
    if final_capital <= initial_capital * 0.5:
        warnings_list.append("🔻 سرمایه شما بیش از ۵۰٪ کاهش یافته")
    
    if final_capital <= 0:
        warnings_list.append("💥 سرمایه شما به صفر رسید")

    return {
        'required_trades': required_trades,
        'reached_target': reached_target,
        'final_capital': final_capital,
        'target_capital': target_capital_value,
        'total_profit': total_profit,
        'total_profit_percent': total_profit_percent,
        'risk_reward_ratio': risk_reward_ratio,
        'effective_win_rate': effective_win_rate,
        'max_drawdown': max_drawdown,
        'expected_return_per_trade': expected_return * 100,
        'stability_status': stability_status,
        'estimated_time': estimated_time,
        'success_probability': success_probability,
        'capital_growth_chart': capital_history,
        'drawdown_chart': drawdown_history,
        'warnings': warnings_list
    }

def generate_recommendations(results, win_rate_percent, profit_per_trade_percent, loss_per_trade_percent):
    """
    تولید توصیه‌های هوشمند بر اساس نتایج
    """
    recommendations = []
    
    # تحلیل سودآوری
    if results['expected_return_per_trade'] <= 0:
        recommendations.append("🎯 **استراتژی خود را بازبینی کنید**: سود مورد انتظار منفی است. یا وین‌ریت را افزایش دهید یا نسبت سود به ضرر را بهبود بخشید.")
    
    # تحلیل وین‌ریت
    if win_rate_percent < results['effective_win_rate']:
        recommendations.append(f"📊 **وین‌ریت را افزایش دهید**: برای سودآوری نیاز به وین‌ریت حداقل {results['effective_win_rate']:.1f}% دارید.")
    elif win_rate_percent < 50:
        recommendations.append("🎯 **وین‌ریت را بهبود بخشید**: وین‌ریت زیر ۵۰٪ ریسک بالایی دارد. هدف خود را حداقل ۵۵-۶۰٪ قرار دهید.")
    
    # تحلیل نسبت R/R
    if results['risk_reward_ratio'] < 1.2:
        recommendations.append("⚖️ **نسبت R/R را بهبود بخشید**: نسبت فعلی پایین است. سعی کنید نسبت را به ۱.۵ یا بالاتر برسانید.")
    elif results['risk_reward_ratio'] > 3:
        recommendations.append("⚖️ **نسبت R/R را واقع‌بینانه کنید**: نسبت بسیار بالا ممکن است غیرواقعی باشد.")
    
    # تحلیل افت سرمایه
    if results['max_drawdown'] > 20:
        recommendations.append("📉 **مدیریت ریسک را تقویت کنید**: افت سرمایه بالا است. حجم معاملات را کاهش دهید یا حد ضرر را کوچک‌تر کنید.")
    
    # تحلیل سود مورد انتظار
    if 0 < results['expected_return_per_trade'] < 0.3:
        recommendations.append("💹 **بازدهی را افزایش دهید**: سود مورد انتظار هر معامله پایین است. استراتژی معاملاتی خود را بازبینی کنید.")
    
    # تحلیل پایداری
    if "ناپایدار" in results['stability_status']:
        recommendations.append("🛡️ **بر روی پایداری تمرکز کنید**: استراتژی شما ناپایدار است. پارامترها را تنظیم و بکتست بیشتری انجام دهید.")
    
    # تحلیل هدف سود
    if not results['reached_target']:
        recommendations.append("🎯 **هدف سود را تعدیل کنید**: با پارامترهای فعلی رسیدن به هدف دشوار است. هدف کوچک‌تری در نظر بگیرید.")
    
    # توصیه‌های عمومی
    if results['success_probability'] > 70:
        recommendations.append("✅ **استراتژی مطلوب**: پارامترهای شما در محدوده مناسب قرار دارند. به همین روال ادامه دهید.")
    else:
        recommendations.append("🔍 **بکتست بیشتری انجام دهید**: قبل از معامله واقعی، استراتژی خود را بیشتر تست کنید.")

    return recommendations

def format_currency(amount):
    """فرمت کردن اعداد به صورت مالی"""
    return f"{amount:,.0f}"

def main():
    st.markdown('<div class="main-header"><h1>🎯 سیستم مدیریت ریسک و سرمایه پیشرفته</h1></div>', unsafe_allow_html=True)
    st.markdown("---")
    
    # سایدبار برای ورودی‌ها
    with st.sidebar:
        st.markdown('<div class="sidebar-header"><h2>📋 پارامترهای ورودی</h2></div>', unsafe_allow_html=True)
        
        initial_capital = st.number_input(
            "💰 سرمایه اولیه (تومان)",
            min_value=1000000,
            value=10000000,
            step=1000000,
            help="سرمایه اولیه شما برای شروع معاملات"
        )
        
        target_profit_percent = st.number_input(
            "🎯 درصد سود هدف",
            min_value=1,
            value=10,
            step=1,
            help="درصد سود کلی که می‌خواهید به آن برسید"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            profit_per_trade_percent = st.number_input(
                "📈 سود هر معامله (%)",
                min_value=0.1,
                value=5.0,
                step=0.1,
                help="میانگین سود در معاملات موفق"
            )
        with col2:
            loss_per_trade_percent = st.number_input(
                "📉 ضرر هر معامله (%)",
                min_value=0.1,
                value=3.0,
                step=0.1,
                help="میانگین ضرر در معاملات ناموفق"
            )
        
        win_rate_percent = st.slider(
            "⚖️ وین‌ریت (%)",
            min_value=0,
            max_value=100,
            value=70,
            help="درصد معاملات موفق شما"
        )
        
        compound_type = st.selectbox(
            "🔁 نوع محاسبه سود",
            ["مرکب", "غیرمرکب"],
            help="مرکب: سود روی سرمایه اضافه می‌شود | غیرمرکب: سود جدا محاسبه می‌شود"
        )
        
        st.markdown("---")
        st.markdown('<div class="sidebar-header"><h3>⚙️ تنظیمات پیشرفته</h3></div>', unsafe_allow_html=True)
        
        max_trades = st.number_input(
            "⏱ حداکثر تعداد معاملات",
            min_value=1,
            value=100,
            step=1,
            help="حداکثر تعداد معاملاتی که انجام می‌دهید"
        )
        
        trades_per_period = st.number_input(
            "📅 تعداد معاملات در هر دوره",
            min_value=1,
            value=10,
            step=1,
            help="تعداد معاملات در هر بازه زمانی (روز/هفته/ماه)"
        )
        
        min_capital = st.number_input(
            "💸 حداقل سرمایه (تومان)",
            min_value=0,
            value=8000000,
            step=1000000,
            help="حداقل سرمایه برای ادامه معاملات"
        )
        
        commission_percent = st.number_input(
            "💳 کارمزد هر معامله (%)",
            min_value=0.0,
            value=0.1,
            step=0.01,
            help="درصد کارمزد هر معامله"
        )
    
    # دکمه محاسبه
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
                trades_per_period=trades_per_period,
                min_capital=min_capital,
                commission_percent=commission_percent
            )
            
            # تولید توصیه‌ها
            recommendations = generate_recommendations(
                results, win_rate_percent, profit_per_trade_percent, loss_per_trade_percent
            )
        
        # نمایش نتایج
        st.markdown("---")
        st.markdown('<div class="main-header"><h2>📊 نتایج تحلیل</h2></div>', unsafe_allow_html=True)
        
        # کارت‌های اصلی
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "🔢 تعداد معاملات لازم",
                f"{results['required_trades']}",
                f"{'✅' if results['reached_target'] else '❌'}"
            )
        
        with col2:
            profit_color = "normal" if results['total_profit'] >= 0 else "inverse"
            st.metric(
                "💰 سرمایه نهایی",
                f"{format_currency(results['final_capital'])} تومان",
                f"{results['total_profit_percent']:+.1f}%",
                delta_color=profit_color
            )
        
        with col3:
            st.metric(
                "⚖️ نسبت R/R",
                f"{results['risk_reward_ratio']:.2f}",
                "مناسب" if results['risk_reward_ratio'] > 1.2 else "نیاز بهبود"
            )
        
        with col4:
            st.metric(
                "🧱 وضعیت پایداری",
                results['stability_status']
            )
        
        # ردیف دوم کارت‌ها
        col5, col6, col7, col8 = st.columns(4)
        
        with col5:
            status_color = "normal" if win_rate_percent >= results['effective_win_rate'] else "inverse"
            st.metric(
                "🎯 وین‌ریت مؤثر",
                f"{results['effective_win_rate']:.1f}%",
                f"{'✅' if win_rate_percent >= results['effective_win_rate'] else '❌'}",
                delta_color=status_color
            )
        
        with col6:
            drawdown_color = "normal" if results['max_drawdown'] < 15 else "inverse"
            st.metric(
                "📉 حداکثر دراودان",
                f"{results['max_drawdown']:.1f}%",
                "کم" if results['max_drawdown'] < 15 else "زیاد",
                delta_color=drawdown_color
            )
        
        with col7:
            expected_color = "normal" if results['expected_return_per_trade'] > 0 else "inverse"
            st.metric(
                "📈 سود مورد انتظار هر معامله",
                f"{results['expected_return_per_trade']:.3f}%",
                delta_color=expected_color
            )
        
        with col8:
            success_color = "normal" if results['success_probability'] > 60 else "inverse"
            st.metric(
                "🎯 احتمال موفقیت",
                f"{results['success_probability']:.1f}%",
                delta_color=success_color
            )
        
        # بخش تحلیل و توصیه
        st.markdown("---")
        st.markdown('<div class="analysis-box">', unsafe_allow_html=True)
        st.subheader("📈 تحلیل پارامترهای ورودی")
        
        col9, col10 = st.columns(2)
        
        with col9:
            st.write(f"**💰 سرمایه اولیه:** {format_currency(initial_capital)} تومان")
            st.write(f"**🎯 هدف سود:** {target_profit_percent}%")
            st.write(f"**🎯 سرمایه هدف:** {format_currency(results['target_capital'])} تومان")
            st.write(f"**📈 سود هر معامله:** {profit_per_trade_percent}%")
        
        with col10:
            st.write(f"**📉 ضرر هر معامله:** {loss_per_trade_percent}%")
            st.write(f"**⚖️ وین‌ریت:** {win_rate_percent}%")
            st.write(f"**🔁 نوع محاسبه:** {compound_type}")
            st.write(f"**⏱ حداکثر معاملات:** {max_trades}")
        
        # تحلیل سریع
        st.markdown("---")
        st.subheader("🧮 تحلیل سریع استراتژی")
        
        expected_return = results['expected_return_per_trade']
        if expected_return > 0:
            st.success(f"**سود مورد انتظار هر معامله: {expected_return:.3f}%** - استراتژی شما از نظر تئوری سودآور است")
        else:
            st.error(f"**سود مورد انتظار هر معامله: {expected_return:.3f}%** - استراتژی شما از نظر تئوری سودآور نیست")
        
        if win_rate_percent >= results['effective_win_rate']:
            st.success(f"**وین‌ریت شما کافی است** - وین‌ریت فعلی ({win_rate_percent}%) از حداقل مورد نیاز ({results['effective_win_rate']:.1f}%) بیشتر است")
        else:
            st.error(f"**وین‌ریت شما کافی نیست** - برای سودآوری نیاز به وین‌ریت حداقل {results['effective_win_rate']:.1f}% دارید")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # بخش توصیه‌ها
        if recommendations:
            st.markdown('<div class="recommendation-box">', unsafe_allow_html=True)
            st.subheader("💡 توصیه‌های بهبود استراتژی")
            for rec in recommendations:
                st.write(f"• {rec}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # نمودارها
        st.markdown("---")
        st.subheader("📈 نمودارهای تحلیلی")
        
        col11, col12 = st.columns(2)
        
        with col11:
            st.markdown("**📊 رشد سرمایه**")
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(
                y=results['capital_growth_chart'],
                mode='lines+markers',
                name='سرمایه',
                line=dict(color='#00cc96', width=3),
                marker=dict(size=4)
            ))
            fig1.add_hline(y=results['target_capital'], line_dash="dash", line_color="red", 
                          annotation_text="هدف سود")
            fig1.add_hline(y=initial_capital, line_dash="dash", line_color="blue", 
                          annotation_text="سرمایه اولیه")
            fig1.update_layout(
                xaxis_title='تعداد معاملات',
                yaxis_title='سرمایه (تومان)',
                height=400,
                showlegend=False
            )
            st.plotly_chart(fig1, use_container_width=True)
        
        with col12:
            st.markdown("**📉 افت سرمایه (Drawdown)**")
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                y=results['drawdown_chart'],
                mode='lines',
                name='دراودان',
                line=dict(color='#ef553b', width=3),
                fill='tozeroy'
            ))
            fig2.add_hline(y=20, line_dash="dash", line_color="red", 
                          annotation_text="حد مجاز افت")
            fig2.update_layout(
                xaxis_title='تعداد معاملات',
                yaxis_title='دراودان (%)',
                height=400,
                showlegend=False
            )
            st.plotly_chart(fig2, use_container_width=True)
        
        # هشدارها
        st.markdown("---")
        if results['warnings']:
            st.markdown('<div class="warning-box">', unsafe_allow_html=True)
            st.subheader("⚠️ هشدارها و نکات مهم")
            for warning in results['warnings']:
                st.write(f"• {warning}")
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="success-box">', unsafe_allow_html=True)
            st.subheader("✅ وضعیت مطلوب")
            st.write("• تمام پارامترها در محدوده مناسب هستند")
            st.write("• احتمال موفقیت بالا است")
            st.write("• افت سرمایه تحت کنترل است")
            st.write("• استراتژی از نظر تئوری سودآور است")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # خلاصه نهایی
        st.markdown("---")
        st.subheader("📋 خلاصه نهایی")
        
        if results['reached_target']:
            st.success(f"""
            🎉 **موفقیت آمیز!** 
            
            - با **{results['required_trades']} معامله** به هدف **{target_profit_percent}%** سود رسیدید
            - سرمایه شما از **{format_currency(initial_capital)}** تومان به **{format_currency(results['final_capital'])}** تومان رسید
            - سود کل: **{format_currency(results['total_profit'])}** تومان (**{results['total_profit_percent']:+.1f}%**)
            """)
        else:
            st.warning(f"""
            ⚠️ **به هدف نرسیدید!** 
            
            - پس از **{results['required_trades']} معامله**، به {target_profit_percent}% سود نرسیدید
            - سرمایه شما به **{format_currency(results['final_capital'])}** تومان رسید
            - سود/ضرر کل: **{format_currency(results['total_profit'])}** تومان (**{results['total_profit_percent']:+.1f}%**)
            - سرمایه هدف: **{format_currency(results['target_capital'])}** تومان
            """)

    else:
        # صفحه اولیه وقتی هنوز محاسبه نشده
        st.markdown("""
        ## 📊 به سیستم مدیریت ریسک و سرمایه خوش آمدید
        
        این برنامه به شما کمک می‌کند:
        
        - 🔍 **تعداد معاملات لازم** برای رسیدن به هدف سود را محاسبه کنید
        - ⚠️ **ریسک‌های احتمالی** را شناسایی کنید  
        - 📈 **پایداری استراتژی** معاملاتی خود را تحلیل کنید
        - 💰 **مدیریت سرمایه** بهینه‌ای داشته باشید
        - 💡 **توصیه‌های هوشمند** برای بهبود استراتژی دریافت کنید
        
        ### 🚀 چگونه شروع کنیم؟
        1. پارامترهای ورودی در سایدبار سمت چپ را پر کنید
        2. روی دکمه "محاسبه نتایج" کلیک کنید
        3. نتایج تحلیل را مشاهده و استراتژی خود را بهینه کنید
        
        ### 💡 نکات مهم:
        - وین‌ریت واقع‌بینانه‌ای وارد کنید
        - حد ضرر منطقی تعیین کنید  
        - کارمزد معاملات را در نظر بگیرید
        - حداقل سرمایه برای ادامه معاملات را مشخص کنید
        
        ### 🔧 قابلیت‌های جدید:
        - ✅ محاسبات **پایدار** و **قابل اطمینان**
        - ✅ پشتیبانی از **مرکب** و **غیرمرکب**
        - ✅ تحلیل **سودآوری استراتژی**
        - ✅ محاسبه **حداقل وین‌ریت مورد نیاز**
        - ✅ **نتایج یکسان** در هر بار محاسبه
        """)

if __name__ == "__main__":
    main()
```
