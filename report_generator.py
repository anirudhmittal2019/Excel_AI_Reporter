import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from transformers import pipeline
from datetime import datetime
import numpy as np
import warnings
import os
warnings.filterwarnings('ignore')

# ====== SETTINGS ======
EXCEL_FILE = "sample_data.xlsx"  # Can be CSV or Excel
PDF_FILE = "analysis_report.pdf"
DATE_FORMATS = ['%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y', '%Y%m%d']  # Add more formats as needed

# ====== LLM Summarizer ======
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def detect_date_column(df):
    """Try to identify which column contains dates"""
    for col in df.columns:
        # Skip if few unique values (probably not date)
        if df[col].nunique() < 10:
            continue
            
        for fmt in DATE_FORMATS:
            try:
                # Try to convert a sample
                pd.to_datetime(df[col].head(10), format=fmt)
                return col
            except:
                continue
    return None

def detect_numeric_columns(df):
    """Identify numeric columns (int or float)"""
    numeric_cols = []
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(col)
    return numeric_cols

def find_most_correlated_columns(df, numeric_cols):
    """Find the top 3 most correlated numeric columns"""
    if len(numeric_cols) < 2:
        return []
    
    corr_matrix = df[numeric_cols].corr().abs()
    np.fill_diagonal(corr_matrix.values, 0)  # Ignore self-correlation
    
    # Get top 3 correlations
    top_correlations = corr_matrix.unstack().sort_values(ascending=False).drop_duplicates()
    top_pairs = top_correlations.head(3).index.tolist()
    
    return top_pairs

import pandas as pd

def generate_insights(df, date_col, numeric_cols):
    """Generate weekly insights based on the data"""
    insights = []
    
    if date_col:
        # Ensure date column is datetime
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df = df.sort_values(date_col)
        
        # Set date as index
        df_time = df.set_index(date_col)
        
        for col in numeric_cols:
            if col == date_col:
                continue
            
            # Calculate weekly aggregates
            weekly = df_time[col].resample('W').mean()
            
            if len(weekly) > 1:
                wow_changes = weekly.pct_change() * 100
                avg_wow_change = wow_changes.mean()
                
                # Average week-on-week growth or decline (always added)
                factor = (weekly.iloc[-1] / weekly.iloc[0]) if weekly.iloc[0] != 0 else float('inf')
                direction = "increase" if factor > 1 else "decrease"
                insights.append(
                    f"📊 On average, there has been a {factor:.2f}x {direction} in {col} week-on-week"
                )
            
            # Outlier detection (still useful)
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            low_threshold = q1 - 1.5 * iqr
            high_threshold = q3 + 1.5 * iqr
            
            low_periods = df[df[col] < low_threshold]
            high_periods = df[df[col] > high_threshold]
            
            if not low_periods.empty:
                start = low_periods[date_col].min().strftime('%Y-%m-%d')
                end = low_periods[date_col].max().strftime('%Y-%m-%d')
                insights.append(f"⚠️ Low {col} detected from {start} to {end} (values below {low_threshold:.2f})")
                insights.append("   - Possible causes: Seasonality, technical issues, or market changes")
                
            if not high_periods.empty:
                start = high_periods[date_col].min().strftime('%Y-%m-%d')
                end = high_periods[date_col].max().strftime('%Y-%m-%d')
                insights.append(f"🚀 High {col} detected from {start} to {end} (values above {high_threshold:.2f})")
                insights.append("   - Possible causes: Promotions, events, or viral growth")
    
    return insights

def generate_summary(df, date_col, numeric_cols, insights):
    """Generate summary using LLM"""
    # Create plain-text description of data
    text = f"This dataset contains {len(df)} records"
    
    if date_col:
        min_date = df[date_col].min()
        max_date = df[date_col].max()
        text += f" from {min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}."
    else:
        text += "."
    
    text += f" The numeric columns are: {', '.join(numeric_cols)}. "
    
    for col in numeric_cols:
        text += f"Column '{col}' has average value {df[col].mean():.2f} (median: {df[col].median():.2f}), ranging from {df[col].min():.2f} to {df[col].max():.2f}. "
    
    if insights:
        text += "Key insights: " + " ".join([insight for insight in insights if not insight.startswith("   -")])
    
    # Summarize with LLM
    try:
        summary = summarizer(text, max_length=200, min_length=80, do_sample=False)[0]['summary_text']
        return summary
    except:
        return "AI summary unavailable. Here are the key stats:\n" + text

def generate_time_series_plot(df, date_col, numeric_cols):
    """Generate time series plots for numeric columns"""
    plot_files = []
    
    if not date_col or not numeric_cols:
        return plot_files
    
    for col in numeric_cols:
        plt.figure(figsize=(10, 5))
        
        # Plot actual values
        sns.lineplot(data=df, x=date_col, y=col, label='Actual')
        
        # Plot rolling average
        rolling_avg = df[col].rolling(window=7, min_periods=1).mean()
        sns.lineplot(x=df[date_col], y=rolling_avg, label='7-day Avg', linestyle='--')
        
        plt.title(f"Time Series of {col}")
        plt.ylabel(col)
        plt.xlabel("Date")
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        filename = f"timeseries_{col}.png"
        plt.savefig(filename, bbox_inches='tight')
        plt.close()
        plot_files.append(filename)
    
    return plot_files

def generate_correlation_plots(df, correlated_pairs):
    """Generate plots for correlated columns"""
    plot_files = []
    
    for i, (col1, col2) in enumerate(correlated_pairs):
        plt.figure(figsize=(8, 6))
        sns.regplot(data=df, x=col1, y=col2, scatter_kws={'alpha':0.5})
        plt.title(f"Correlation between {col1} and {col2}")
        
        # Calculate correlation coefficient
        corr = df[[col1, col2]].corr().iloc[0,1]
        plt.xlabel(f"{col1} (r = {corr:.2f})")
        plt.ylabel(col2)
        
        filename = f"correlation_{i}.png"
        plt.savefig(filename, bbox_inches='tight')
        plt.close()
        plot_files.append(filename)
    
    return plot_files

def generate_distribution_plots(df, numeric_cols):
    """Generate distribution plots for numeric columns"""
    plot_files = []
    
    for col in numeric_cols:
        plt.figure(figsize=(8, 5))
        
        # Histogram with KDE
        sns.histplot(data=df, x=col, kde=True)
        
        # Add vertical lines for mean and median
        plt.axvline(df[col].mean(), color='red', linestyle='--', label=f'Mean: {df[col].mean():.2f}')
        plt.axvline(df[col].median(), color='green', linestyle=':', label=f'Median: {df[col].median():.2f}')
        
        plt.title(f"Distribution of {col}")
        plt.legend()
        plt.tight_layout()
        
        filename = f"distribution_{col}.png"
        plt.savefig(filename, bbox_inches='tight')
        plt.close()
        plot_files.append(filename)
    
    return plot_files

def generate_report():
    # Read data
    try:
        if EXCEL_FILE.endswith('.csv'):
            df = pd.read_csv(EXCEL_FILE)
        else:
            df = pd.read_excel(EXCEL_FILE)
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    # Clean data
    df = df.dropna(how='all').dropna(axis=1, how='all')
    
    # Detect columns
    date_col = detect_date_column(df)
    numeric_cols = detect_numeric_columns(df)
    correlated_pairs = find_most_correlated_columns(df, numeric_cols)
    
    # Generate content
    insights = generate_insights(df, date_col, numeric_cols)
    time_series_plots = generate_time_series_plot(df, date_col, numeric_cols) if date_col else []
    correlation_plots = generate_correlation_plots(df, correlated_pairs[:3]) if correlated_pairs else []
    distribution_plots = generate_distribution_plots(df, numeric_cols) if numeric_cols else []
    ai_summary = generate_summary(df, date_col, numeric_cols, insights)

    # ===== Null Value Summary =====
    null_summary = df.isnull().sum()
    null_summary = null_summary[null_summary > 0]

    if not null_summary.empty:
        insights.append("📌 Missing Data Detected:")
        for col, count in null_summary.items():
            percent = (count / len(df)) * 100
            insights.append(f"   - Column '{col}' has {count} missing values ({percent:.1f}%)")
    
    # ====== PDF Generation ======
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(PDF_FILE, pagesize=A4)
    story = []

    # Title
    story.append(Paragraph("Data Analysis Report", styles["Title"]))
    story.append(Spacer(1, 24))
    
    # 1. Data Preview Section
    story.append(Paragraph("<b>1. Data Preview</b>", styles["Heading1"]))
    story.append(Spacer(1, 12))
    
    # Basic stats
    stats_text = f"• Total records: {len(df)}"
    if date_col:
        min_date = df[date_col].min()
        max_date = df[date_col].max()
        stats_text += f"<br/>• Date range: {min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}"
    stats_text += f"<br/>• Numeric columns: {', '.join(numeric_cols)}"
    story.append(Paragraph(stats_text, styles["BodyText"]))
    story.append(Spacer(1, 12))
    
    # Data table
    story.append(Paragraph("<b>Sample Data:</b>", styles["Heading2"]))
    table_data = [df.columns.tolist()] + df.head(10).values.tolist()
    table = Table(table_data)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#2c3e50")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.whitesmoke),
        ("ALIGN", (0,0), (-1,-1), "CENTER"),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE", (0,0), (-1,0), 9),
        ("BOTTOMPADDING", (0,0), (-1,0), 12),
        ("BACKGROUND", (0,1), (-1,-1), colors.HexColor("#ecf0f1")),
        ("GRID", (0,0), (-1,-1), 0.5, colors.HexColor("#bdc3c7")),
        ("FONTSIZE", (0,1), (-1,-1), 8)
    ]))
    story.append(table)
    story.append(Spacer(1, 24))
    
    # 2. Key Insights Section
    story.append(Paragraph("<b>2. Key Insights</b>", styles["Heading1"]))
    story.append(Spacer(1, 12))
    
    if insights:
        for insight in insights:
            if insight.startswith("   -"):
                # Sub-bullet point
                story.append(Paragraph(f"&nbsp;&nbsp;&nbsp;&nbsp;{insight[4:]}", styles["BodyText"]))
            else:
                # Main bullet point
                story.append(Paragraph(f"• {insight}", styles["BodyText"]))
    else:
        story.append(Paragraph("No significant patterns detected automatically.", styles["BodyText"]))
    story.append(Spacer(1, 24))
    
    # 3. Visualizations Section
    story.append(Paragraph("<b>3. Visualizations</b>", styles["Heading1"]))
    story.append(Spacer(1, 12))
    
    # Time series plots
    if time_series_plots:
        story.append(Paragraph("<b>Time Series Analysis:</b>", styles["Heading2"]))
        for plot_file in time_series_plots:
            story.append(Image(plot_file, width=500, height=300))
            story.append(Spacer(1, 12))
    
    # Distribution plots
    if distribution_plots:
        story.append(Paragraph("<b>Data Distributions:</b>", styles["Heading2"]))
        for plot_file in distribution_plots:
            story.append(Image(plot_file, width=500, height=300))
            story.append(Spacer(1, 12))
    
    # Correlation plots
    if correlation_plots:
        story.append(Paragraph("<b>Top Correlations:</b>", styles["Heading2"]))
        for plot_file in correlation_plots:
            story.append(Image(plot_file, width=500, height=350))
            story.append(Spacer(1, 12))
    
    # 4. Summary Report Section
    story.append(Paragraph("<b>4. Summary Report</b>", styles["Heading1"]))
    story.append(Spacer(1, 12))
    
    story.append(Paragraph(ai_summary, styles["BodyText"]))
    story.append(Spacer(1, 12))
    
    doc.build(story)
    
    # Clean up temporary files
    all_plots = time_series_plots + correlation_plots + distribution_plots
    for plot_file in all_plots:
        try:
            os.remove(plot_file)
        except:
            pass
    
    print(f"✅ Report generated: {PDF_FILE}")

# if __name__ == "__main__":
#     generate_report()