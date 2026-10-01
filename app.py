from __future__ import annotations
from pathlib import Path
import sys
import numpy as np 
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st 

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0,str(PROJECT_ROOT))
from src.utils import find_similar_customers ,load_artifact , load_json , recommended_action

MODELS_DIR =  PROJECT_ROOT /'models'
CLUSTERED_DATA_PATH =MODELS_DIR/'clustered_customers.csv'
PROFILE_PATH = MODELS_DIR/'segment_profiles.csv'
DIFF_K_PATH = MODELS_DIR/'diff_k_scores.csv'
COMPARISON_PATH = MODELS_DIR/'model_comparison.csv'

st.set_page_config(page_title='Customer Segmentation Intelligence',page_icon='📊',layout='wide')
@st.cache_data
def load_data() :
    """Load trained customer data."""
    if not CLUSTERED_DATA_PATH.exists(): raise FileNotFoundError('Run `python train_pipeline.py` first.')
    return pd.read_csv(CLUSTERED_DATA_PATH)

@st.cache_data 
def load_profiles() :
    """Load segment profiles."""
    if not PROFILE_PATH.exists() : raise FileNotFoundError('Run the training pipeline first.')
    return pd.read_csv(PROFILE_PATH)
@st.cache_data
def load_elbow() :
    """Load K selection metrics."""
    return pd.read_csv(DIFF_K_PATH) if DIFF_K_PATH.exists() else pd.DataFrame()
@st.cache_data 
def load_comparison() :
    """Load clustering comparison."""
    return pd.read_csv(COMPARISON_PATH) if COMPARISON_PATH.exists() else pd.DataFrame()

@st.cache_resource
def load_models() :
    """Load serialized ML artifacts."""
    return {'scaler' : load_artifact(str(MODELS_DIR/'scaler.joblib')),'kmeans' : load_artifact(str(MODELS_DIR/'kmeans_model.joblib')),'pca':load_artifact(str(MODELS_DIR/'pca_model.joblib')),'similarity_matrix' :load_artifact(str(MODELS_DIR/'similarity_matrix.joblib')),'metadata' : load_json(str(MODELS_DIR/'metadata.json'))}

def insight_box(title:str,insights:list[str],conclusion :str) :
    """Render evidence and a separate business conclusion column."""
    st.markdown('---') ; st.subheader(f'💡 {title}')
    a,b = st.columns([2,1])
    with a :
        st.markdown('**What the data tells us**')
        for item in insights : st.markdown(f'- {item}')
    with b :
        st.markdown('**Conclusion / Buisness Meaning**'); st.info(conclusion)

def money(x:float) :
    return f'₹{x:,.0f}'

def overview(df:pd.DataFrame,profiles :pd.DataFrame) :
    """Render executive overview with attractive business charts."""
    st.header('📊 Customer Segmentation Overview')
    st.caption('Customer scale, value and segment distribution.')
    a,b,c,d =st.columns(4) 
    a.metric('Total Customers',f'{len(df):,}') 
    b.metric('Total Customer Purchase',money(df.MonetaryValue.sum()))
    c.metric('Average Customer Purchase',money(df.MonetaryValue.mean()))
    d.metric('Avg. Purchase Frequency' , f'{df.Frequency.mean():.1f}')
    x = profiles.sort_values('CustomerCount').copy()
    x['Segment'] = 'Cluster' + x.Cluster.astype(str)+' — '+x.SegmentLabel

    fig = px.bar(x,x='CustomerCount',y='Segment',orientation='h',color='SegmentLabel',
                 text='CustomerCount',color_discrete_sequence=px.colors.qualitative.Bold,
                 title='Customer Distribution Across Behavioral Segments',labels={'CustomerCount':'Customers','Segment':''})
    fig.update_traces(textposition = 'outside')
    fig.update_layout(height=480,showlegend=False,plot_bgcolor='rgba(0,0,0,0)',paper_bgcolor = 'rgba(0,0,0,0)')
    st.plotly_chart(fig,use_container_width=True)
    st.caption('Longer bars mean larger customer groups; the business label translates the ML cluster into an actionable segment.')
    v = profiles.sort_values('AverageMonetaryValue',ascending=False).copy() 
    v['Segment']='Cluster '+v.Cluster.astype(str)+' — '+v.SegmentLabel
    fig=px.bar(v,x='Segment',y='AverageMonetaryValue',color='SegmentLabel',text_auto='.2s',
               color_discrete_sequence=px.colors.qualitative.Set2,
               title='Average Customer Value by Segment',labels={'AverageMonetaryValue':'Average Customer Purchase (₹)','Segment':''})
    fig.update_layout(height=450,showlegend=False,plot_bgcolor='rgba(0,0,0,0)',paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig,use_container_width=True)
    largest=profiles.loc[profiles.CustomerCount.idxmax()]
    highest=profiles.loc[profiles.AverageMonetaryValue.idxmax()]
    insight_box('Overview Insights',[f"The largest segment is **{largest.SegmentLabel}** with {int(largest.CustomerCount)} customers ({largest.Percentage:.1f}%).",
                                     f"**{highest.SegmentLabel}** has the highest average customer purchase at {money(highest.AverageMonetaryValue)}.",
                                     'Segment size and segment value are different business dimensions.'],
                                     'Prioritize campaigns using both reach and economic value; the largest segment is not automatically the most valuable.')


def eda(df:pd.DataFrame) :
    """Render interactive behavioral EDA."""
    st.header('🔎 Behavioral EDA')
    st.caption('Explore relationships among the behavioral features used by the clustering model.')
    features = ['Recency' , 'Frequency' , 'MonetaryValue' ,'AOV','TotalQuantityPurchased','UniqueProducts','UniqueCategories' ,'AverageDaysBetweenPurchases']
    a,b = st.columns(2) 
    x =a.selectbox('X-axis feature',features)
    y= b.selectbox('Y-axis feature',features,index=2)

    fig = px.scatter(df,x=x,y=y,color=df.Cluster.astype(str),opacity=.72,color_discrete_sequence=px.colors.qualitative.Bold,hover_data=['CustomerID' ,'Recency','Frequency','MonetaryValue','AOV'],
                     title = f"{x} vs {y} : Customer Behaviour")
    fig.update_layout(height=560,plot_bgcolor='rgba(0,0,0,0)',paper_bgcolor ='rgba(0,0,0,0)',legend_title='Cluster')
    st.plotly_chart(fig,use_container_width=True)

    feature = st.selectbox('Compare feature across clusters' ,features,index=2)
    fig = px.box(df,x=df.Cluster.astype(str),y=feature,color= df.Cluster.astype(str),color_discrete_sequence=px.colors.qualitative.Vivid,title=f'{feature} Distribution by Cluster')
    fig.update_layout(height=470,showlegend=False,plot_bgcolor='rgba(0,0,0,0)',paper_bgcolor ='rgba(0,0,0,0)')
    st.plotly_chart(fig,use_container_width=True)

    corr = df[features].corr()
    arr = corr.where(~np.eye(len(corr),dtype=bool)).stack().abs().sort_values(ascending=False); pair=arr.index[0] if len(arr) else ('selected','features')
    insight_box('EDA Insights & Conclusion',[f'The selected scatter plot show how **{x}** and **{y}** vary across clusters.',
                f'The box plot highlight differences in **{feature}** ,including spread and outliers.',
                f'The strongest pairwise relationship in the selected feature set is **{pair[0]}** and **{pair[1]}**.'],
                'Use these relationships to understand why segments differ and to formulate testable marketing hypotheses;correlation does not prove causation.')

def profiles_page(profiles:pd.DataFrame) :
    """Render segment profiles and business strategies."""
    st.header('🎯 Segment Profiling')
    st.caption('Translate ML clusters into customer groups and strategies.')
    cols = ['Cluster' ,'SegmentLabel','CustomerCount','Percentage','AverageRecency','AverageFrequency','AverageMonetaryValue' ,'AverageAOV' ,'AverageQuantity','AverageUniqueProducts','AverageUniqueCategories','AveragePurchaseGap']
    st.dataframe(profiles[cols].round(2),use_container_width=True,hide_index=True)
    cluster = st.selectbox('Select a cluster for detailed strategy',profiles.Cluster.tolist())
    row = profiles[profiles.Cluster==cluster].iloc[0]
    a,b,c,d = st.columns(4) 
    a.metric('Segment',row.SegmentLabel) 
    b.metric('Customers' ,int(row.CustomerCount))
    c.metric('Avg. Purchase Value',money(row.AverageMonetaryValue))
    d.metric('Avg. Recency' , f'{row.AverageRecency:.0f} days')
    st.markdown('### Recommended Buisness Strategy')
    st.success(recommended_action(row.SegmentLabel))

    insight_box('Segment Insights & Conclusion',[f"This segment represents **{row.Percentage:.1f}%** of customers." ,
                 f"Average frequency is **{row.AverageFrequency:.1f}** transactions." ,f"Average order value is **{money(row.AverageAOV)}**.",f"Customers buy from an average of **{row.AverageUniqueCategories:.1f} categories**."],
                 f"Treat this group as **{row.SegmentLabel}** for campaign planning and validate the recommendation using campaign results.")

def customer_analyzer(df:pd.DataFrame,models:dict) :
    """Render customer-level behavior and similarity search."""
    st.header('👤 Individual Customer Analyzer')
    st.caption('Inspect one customer and their nearest behavioral peers.')
    profiles = load_profiles()
    cid = st.selectbox('Customer ID',sorted(df.CustomerID.astype(str).tolist()))
    selected = df[df.CustomerID.astype(str)==cid]
    if selected.empty: st.error('Customer ID was not found.') ;return
    customer = selected.iloc[0] 
    a,b,c,d = st.columns(4)
    a.metric('Cluster' , int(customer.Cluster))
    b.metric('Recency' , f'{customer.Recency:.0f} days')
    c.metric('Frequency' , f'{customer.Frequency:.0f}')
    d.metric('Customer Spend Value',money(customer.MonetaryValue))

    p = profiles[profiles.Cluster==customer.Cluster]
    label = p.iloc[0].SegmentLabel if not p.empty else 'Unknown'
    st.success(f'Business Segment: **{label}**')
    st.info(recommended_action(label))
    top = st.slider('Similar customers',3,15,5)
    try :
        sim = find_similar_customers(cid,df,models['similarity_matrix'],top)
        sim = sim.merge(df[['CustomerID','Cluster']].astype({'CustomerID':str}),on='CustomerID').merge(profiles[['Cluster' , 'SegmentLabel']],on='Cluster')
        fig = px.bar(sim.sort_values('Distance'),x='Distance',y='CustomerID',orientation='h',color='SegmentLabel',text_auto = '.3f',color_discrete_sequence=px.colors.qualitative.Bold,title='Nearest Behavioral Neighbors — Lower Distance Is Better')
        fig.update_layout(height=430,plot_bgcolor='rgba(0,0,0,0)',paper_bgcolor='rgba(0,0,0,0)'); st.plotly_chart(fig,use_container_width=True); st.dataframe(sim.round(4),use_container_width=True,hide_index=True)
        insight_box('Customer-Level Insights & Conclusion',[f'Customer **{cid}** belongs to cluster **{int(customer.Cluster)}**.',f'The closest displayed behavioral neighbor has distance **{sim.Distance.min():.3f}**.','Similarity uses Euclidean distance in the standardized feature space.'],f'Use the segment and nearest behavioral peers to personalize retention, cross-sell or reactivation actions for **{label}**.')
    except ValueError as exc: st.error(str(exc))

def pca_page(df:pd.DataFrame , models:dict) :
    """Render 2D/3D PCA visualization and interpretation."""
    st.header('🧬 PCA Visualization')
    st.caption('PCA compresses the behavioral feature space for visual inspection.')
    explained = models['pca'].explained_variance_ratio_
    a,b=st.columns(2)
    a.metric('PC1 + PC2' ,f'{explained[:2].sum():.1%}')
    b.metric('Total Shown Variance',f'{explained.sum():.1%}')
    fig = px.scatter(df,x='PC1' ,y = 'PC2',color = df.Cluster.astype(str) ,color_discrete_sequence = px.colors.qualitative.Bold,hover_data=['CustomerID','Recency','Frequency','MonetaryValue'],title = '2D PCA - Customer Segments')
    fig.update_layout(height = 600 ,plot_bgcolor='rgba(0,0,0,0)',paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig,use_container_width=True)
    if 'PC3' in df: st.plotly_chart(px.scatter_3d(df,x='PC1',y='PC2',z='PC3',color=df.Cluster.astype(str),color_discrete_sequence=px.colors.qualitative.Bold,hover_data=['CustomerID','Recency','Frequency','MonetaryValue'],title='3D PCA'),use_container_width=True)
    st.dataframe(pd.DataFrame({'Component':[f'PC{i+1}' for i in range(len(explained))],'Explained Variance (%)':explained*100}).round(2),use_container_width=True,hide_index=True)
    insight_box('PCA Insights & Conclusion',[f'PC1 and PC2 explain **{explained[:2].sum():.1%}** of the standardized variance.',
        'Separated regions suggest stronger behavioral differences between clusters.',
        'Overlap is expected because PCA compresses many dimensions.'],
        'PCA is primarily a visualization tool; the original standardized features remain the basis for clustering and similarity.')

def evaluation_page() :
    """Render K selection and model comparison."""
    st.header('🧪 Model Evaluation')
    st.caption('Validate K-Means and compare alternative clustering algorithms.')
    elbow = load_elbow()
    if not elbow.empty :
        a,b = st.columns(2);
        with a :
            fig = px.line(elbow,x='K',y='Inertia',markers=True,color_discrete_sequence=['#2563EB'],title='Elbow Method — Inertia vs K') 
            fig.update_layout(height=450,plot_bgcolor='rgba(0,0,0,0)',paper_bgcolor = 'rgba(0,0,0,0)')
            st.plotly_chart(fig,use_container_width=True)
        with b :
            fig = px.line(elbow,x='K',y='SilhouetteScore',markers=True,color_discrete_sequence=['#16A34A'],title='Silhouette Score — Higher Is Better')
            fig.update_layout(height= 450 , plot_bgcolor= 'rgba(0,0,0,0)',paper_bgcolor ='rgba(0,0,0,0)')
            st.plotly_chart(fig,use_container_width=True)
    comparison = load_comparison()
    if not comparison.empty :
        st.subheader('K-Means vs Agglomerative vs DBSCAN')
        st.dataframe(comparison.round(4),use_container_width=True,hide_index=True)
        fig = px.bar(comparison, x='Model',y='SilhouetteScore' ,color = 'Model',text_auto='.3f',color_discrete_sequence=px.colors.qualitative.Bold,title='Clustering Quality Comparison')
        fig.update_layout(height=440,showlegend=False,plot_bgcolor='rgba(0,0,0,0)',paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig,use_container_width=True)
        insight_box('Model Evaluation Insights & Conclusion',['Elbow Method Evaluates within-cluster inertia as k changes.',
                    'Silhouette Score evaluates separation and cohesion.',
                    'DBSCAN can identify noise while K-Means and Agglomerative use a chosen cluster count.'],
                    'Select a model using both statistical quality and business interpretability; the highest score alone is not automatically the best business segmentation.')

def main() :
    """Run dashboard."""
    st.title('Customer Segmentation Intelligence Platform')
    st.caption('RFM + Extended Behavioral Features • K-Means • PCA • Similarity Search')
    try :
        df = load_data()
        profiles = load_profiles()
        models = load_models()
    except Exception as exc :
        st.error(str(exc))
        st.info('Run `python train_pipeline.py` first.')
        st.stop()
    tabs = st.tabs(['📊 Overview','🔎 Behavioral EDA','🎯 Segment Profiles','👤 Customer Analyzer','🧬 PCA Visualization','🧪 Model Evaluation'])
    with tabs[0] : overview(df,profiles)
    with tabs[1] : eda(df)
    with tabs[2] : profiles_page(profiles)
    with tabs[3] : customer_analyzer(df,models)
    with tabs[4] : pca_page(df,models)
    with tabs[5] : evaluation_page()

if __name__=='__main__' : 
    main()