# 🚀 Retail Insights Assistant - Scalability Architecture for 100GB+ Data

## Executive Summary

This document outlines a comprehensive architecture for scaling the Retail Insights Assistant to handle 100GB+ datasets efficiently, with enterprise-grade data processing, storage, and retrieval capabilities.

---

## 📊 Current vs. Scalable Architecture Comparison

| **Aspect** | **Current (Small Data)** | **Scalable (100GB+)** |
|------------|-------------------------|----------------------|
| Data Storage | Local CSV files | Cloud Data Warehouse (BigQuery/Snowflake) |
| Processing | Pandas (in-memory) | Apache Spark / Dask (distributed) |
| Search/Retrieval | Full table scans | Vector DB + Semantic Search |
| API | Direct Gemini calls | LangChain with caching |
| Infrastructure | Single server | Kubernetes cluster |
| Cost | Minimal | $500-2000/month |

---

## 🏗️ Scalable Architecture Components

### 1️⃣ DATA INGESTION & PREPROCESSING

#### **Technology Stack:**
- **Apache Spark** (PySpark) - Distributed data processing
- **Apache Airflow** - Workflow orchestration
- **Delta Lake** - ACID transactions on data lakes

#### **Implementation:**

```python
# File: data_ingestion_pipeline.py

from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from delta import *

class ScalableDataPipeline:
    def __init__(self, cloud_provider='gcp'):
        self.spark = (SparkSession.builder
            .appName("RetailInsights")
            .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
            .config("spark.sql.catalog.spark_catalog", 
                   "org.apache.spark.sql.delta.catalog.DeltaCatalog")
            .config("spark.executor.memory", "16g")
            .config("spark.driver.memory", "8g")
            .config("spark.executor.cores", "4")
            .getOrCreate())
    
    def ingest_large_dataset(self, source_path: str, output_path: str):
        """
        Ingest 100GB+ CSV data with partitioning and optimization
        """
        # Read with schema inference and partitioning
        df = (self.spark.read
            .option("header", "true")
            .option("inferSchema", "true")
            .option("mode", "PERMISSIVE")
            .csv(source_path))
        
        # Data cleaning and transformations
        df_cleaned = (df
            .withColumn("Amount", col("Amount").cast("double"))
            .withColumn("Date", to_date(col("Date")))
            .withColumn("Year", year(col("Date")))
            .withColumn("Month", month(col("Date")))
            .withColumn("Quarter", quarter(col("Date")))
            .filter(col("Amount") > 0)
            .dropDuplicates(["Order ID"]))
        
        # Write as Delta Lake with partitioning
        (df_cleaned.write
            .format("delta")
            .mode("overwrite")
            .partitionBy("Year", "Month")
            .option("overwriteSchema", "true")
            .save(output_path))
        
        # Create optimized aggregations
        self._create_aggregated_views(df_cleaned, output_path)
        
        return df_cleaned
    
    def _create_aggregated_views(self, df, output_path):
        """Create pre-aggregated views for faster queries"""
        
        # Monthly aggregates
        monthly_agg = (df.groupBy("Year", "Month", "ship-state")
            .agg(
                sum("Amount").alias("total_revenue"),
                count("Order ID").alias("order_count"),
                avg("Amount").alias("avg_order_value")
            ))
        
        monthly_agg.write.format("delta").mode("overwrite") \
            .save(f"{output_path}/aggregates/monthly")
        
        # Category aggregates
        category_agg = (df.groupBy("Category", "Year", "Quarter")
            .agg(
                sum("Amount").alias("total_revenue"),
                count("Order ID").alias("order_count")
            ))
        
        category_agg.write.format("delta").mode("overwrite") \
            .save(f"{output_path}/aggregates/category")

# Airflow DAG for scheduling
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'data-team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'retries': 3,
    'retry_delay': timedelta(minutes=5)
}

dag = DAG(
    'retail_insights_etl',
    default_args=default_args,
    schedule_interval='@daily',
    catchup=False
)

def run_ingestion():
    pipeline = ScalableDataPipeline()
    pipeline.ingest_large_dataset(
        source_path='gs://retail-data/raw/*.csv',
        output_path='gs://retail-data/processed/'
    )

ingest_task = PythonOperator(
    task_id='ingest_data',
    python_callable=run_ingestion,
    dag=dag
)
```

---

### 2️⃣ CLOUD DATA WAREHOUSE

#### **Recommended: Google BigQuery**

**Why BigQuery?**
- Serverless, auto-scaling
- SQL-based (familiar interface)
- Integrated with Gemini AI
- Cost-effective ($5/TB queried)

#### **Schema Design:**

```sql
-- Main fact table (partitioned by date, clustered by state)
CREATE TABLE `retail_insights.sales_fact`
(
  order_id STRING NOT NULL,
  order_date DATE NOT NULL,
  status STRING,
  fulfillment STRING,
  sales_channel STRING,
  category STRING,
  sku STRING,
  size STRING,
  quantity INT64,
  amount FLOAT64,
  ship_state STRING,
  ship_city STRING,
  ship_postal_code STRING,
  b2b BOOL,
  year INT64,
  month INT64,
  quarter INT64
)
PARTITION BY order_date
CLUSTER BY ship_state, category
OPTIONS(
  description="Main sales transactions table",
  partition_expiration_days=NULL
);

-- Pre-aggregated monthly summary
CREATE MATERIALIZED VIEW `retail_insights.monthly_summary` AS
SELECT 
  DATE_TRUNC(order_date, MONTH) as month,
  ship_state,
  category,
  SUM(amount) as total_revenue,
  COUNT(*) as order_count,
  AVG(amount) as avg_order_value,
  COUNTIF(b2b) as b2b_orders
FROM `retail_insights.sales_fact`
WHERE status != 'Cancelled'
GROUP BY month, ship_state, category;

-- Indexes for fast lookups
CREATE INDEX idx_state_date 
ON `retail_insights.sales_fact`(ship_state, order_date);
```

#### **Python Integration:**

```python
from google.cloud import bigquery
import pandas as pd

class BigQueryConnector:
    def __init__(self, project_id: str):
        self.client = bigquery.Client(project=project_id)
        self.dataset_id = "retail_insights"
    
    def query_with_cache(self, query: str) -> pd.DataFrame:
        """Execute query with results caching"""
        job_config = bigquery.QueryJobConfig(
            use_query_cache=True,
            use_legacy_sql=False
        )
        
        query_job = self.client.query(query, job_config=job_config)
        return query_job.to_dataframe()
    
    def get_summary_stats(self, filters: dict = None) -> dict:
        """Get pre-aggregated stats with optional filters"""
        where_clause = self._build_where_clause(filters)
        
        query = f"""
        SELECT 
            SUM(total_revenue) as revenue,
            SUM(order_count) as orders,
            AVG(avg_order_value) as aov
        FROM `{self.dataset_id}.monthly_summary`
        {where_clause}
        """
        
        result = self.query_with_cache(query)
        return result.iloc[0].to_dict()
```

---

### 3️⃣ SEMANTIC SEARCH & VECTOR DATABASE

#### **Technology: Pinecone + LangChain**

For intelligent data retrieval from massive datasets:

```python
from langchain.embeddings import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.chains import RetrievalQA
from langchain.llms import GoogleGenerativeAI
import pinecone

class SemanticSearchEngine:
    def __init__(self, pinecone_api_key: str, gemini_api_key: str):
        # Initialize Pinecone
        pinecone.init(api_key=pinecone_api_key, environment="us-west1-gcp")
        
        # Create embeddings
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=gemini_api_key
        )
        
        # Connect to vector store
        self.index_name = "retail-insights"
        if self.index_name not in pinecone.list_indexes():
            pinecone.create_index(
                name=self.index_name,
                dimension=768,  # Gemini embedding dimension
                metric="cosine"
            )
        
        self.vectorstore = Pinecone.from_existing_index(
            self.index_name,
            self.embeddings
        )
    
    def index_aggregated_insights(self, bigquery_connector: BigQueryConnector):
        """Index pre-computed insights for semantic search"""
        
        # Get monthly insights
        insights = bigquery_connector.query_with_cache("""
            SELECT 
                CONCAT(
                    'In ', FORMAT_DATE('%B %Y', month), ', ',
                    ship_state, ' generated ₹', 
                    FORMAT('%0.2f', total_revenue),
                    ' revenue from ', order_count, ' orders ',
                    'in ', category, ' category'
                ) as insight_text,
                month, ship_state, category, total_revenue
            FROM `retail_insights.monthly_summary`
            ORDER BY total_revenue DESC
            LIMIT 10000
        """)
        
        # Create documents for vector DB
        texts = insights['insight_text'].tolist()
        metadatas = insights[['month', 'ship_state', 'category', 'total_revenue']].to_dict('records')
        
        self.vectorstore.add_texts(texts, metadatas=metadatas)
    
    def search_similar_insights(self, query: str, k: int = 5):
        """Find most relevant insights for a query"""
        results = self.vectorstore.similarity_search(query, k=k)
        return results

class IntelligentQueryRouter:
    """Routes queries to appropriate data sources"""
    
    def __init__(self, bq_connector, semantic_search, gemini_api_key):
        self.bq = bq_connector
        self.search = semantic_search
        self.llm = GoogleGenerativeAI(
            model="gemini-pro",
            google_api_key=gemini_api_key,
            temperature=0.3
        )
    
    def route_query(self, user_query: str) -> dict:
        """Intelligently route query to best data source"""
        
        # Classify query type
        query_type = self._classify_query(user_query)
        
        if query_type == 'aggregated':
            # Use pre-computed aggregates from BigQuery
            return self._handle_aggregated_query(user_query)
        
        elif query_type == 'semantic':
            # Use vector search for complex natural language
            return self._handle_semantic_query(user_query)
        
        else:
            # Direct SQL for specific data points
            return self._handle_direct_query(user_query)
    
    def _classify_query(self, query: str) -> str:
        """Use LLM to classify query type"""
        prompt = f"""
        Classify this query into one category:
        - aggregated: requests for totals, averages, trends
        - semantic: complex business questions, comparisons
        - direct: specific order/product lookups
        
        Query: {query}
        Classification:"""
        
        response = self.llm(prompt)
        return response.strip().lower()
```

---

### 4️⃣ CACHING & OPTIMIZATION

#### **Redis for Response Caching:**

```python
import redis
import hashlib
import json

class SmartCachingLayer:
    def __init__(self, redis_host='localhost', redis_port=6379):
        self.redis_client = redis.Redis(
            host=redis_host,
            port=redis_port,
            db=0,
            decode_responses=True
        )
        self.cache_ttl = 3600  # 1 hour
    
    def get_cached_response(self, query: str) -> dict:
        """Get cached response for query"""
        cache_key = self._generate_cache_key(query)
        cached = self.redis_client.get(cache_key)
        
        if cached:
            return json.loads(cached)
        return None
    
    def cache_response(self, query: str, response: dict):
        """Cache response with TTL"""
        cache_key = self._generate_cache_key(query)
        self.redis_client.setex(
            cache_key,
            self.cache_ttl,
            json.dumps(response)
        )
    
    def _generate_cache_key(self, query: str) -> str:
        """Generate consistent cache key"""
        return f"query:{hashlib.md5(query.encode()).hexdigest()}"
```

---

### 5️⃣ DEPLOYMENT ARCHITECTURE

#### **Kubernetes Cluster Configuration:**

```yaml
# kubernetes/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: retail-insights-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: retail-insights
  template:
    metadata:
      labels:
        app: retail-insights
    spec:
      containers:
      - name: api
        image: gcr.io/project/retail-insights:latest
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
          limits:
            memory: "8Gi"
            cpu: "4"
        env:
        - name: BIGQUERY_PROJECT
          valueFrom:
            secretKeyRef:
              name: gcp-credentials
              key: project_id
        - name: GEMINI_API_KEY
          valueFrom:
            secretKeyRef:
              name: api-keys
              key: gemini_key
        ports:
        - containerPort: 8080
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: retail-insights-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: retail-insights-api
  minReplicas: 3
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

---

## 💰 Cost Analysis (100GB Data, Monthly)

| **Component** | **Service** | **Specs** | **Cost** |
|--------------|-------------|-----------|----------|
| Data Warehouse | BigQuery | 100GB storage + 1TB queries | $200 |
| Data Processing | Cloud Dataproc (Spark) | 5 nodes, 8 hrs/day | $400 |
| Vector Database | Pinecone | 1M vectors, Standard tier | $70 |
| Caching | Redis Cloud | 5GB | $50 |
| Kubernetes | GKE | 3-20 nodes auto-scaling | $500 |
| AI API | Gemini Pro | 500K requests | $200 |
| Storage | Cloud Storage | 100GB + backups | $30 |
| **Total** | | | **~$1,450/month** |

---

## 📈 Performance Benchmarks

| **Metric** | **Small Data** | **100GB Data** |
|-----------|---------------|---------------|
| Query Response | 2-5 seconds | 0.5-2 seconds |
| Data Load Time | 10 seconds | 5 minutes |
| Concurrent Users | 10-50 | 1000+ |
| Aggregation Speed | 1 second | 0.2 seconds (pre-agg) |

---

## 🔐 Security & Compliance

```python
# security/data_masking.py

class DataSecurityLayer:
    @staticmethod
    def mask_pii(df: pd.DataFrame) -> pd.DataFrame:
        """Mask personally identifiable information"""
        pii_columns = ['ship-postal-code', 'ship-city']
        
        for col in pii_columns:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: 
                    hashlib.sha256(str(x).encode()).hexdigest()[:10]
                )
        
        return df
    
    @staticmethod
    def apply_row_level_security(user_role: str, query: str) -> str:
        """Add row-level security filters"""
        if user_role == 'regional_manager':
            return f"{query} WHERE ship_state IN (SELECT allowed_states FROM user_permissions WHERE user_id = CURRENT_USER())"
        return query
```

---

## 🚀 Migration Path

### Phase 1: Foundation (Week 1-2)
1. Set up BigQuery dataset
2. Migrate historical data with Spark
3. Create partitioned tables

### Phase 2: Optimization (Week 3-4)
1. Build materialized views
2. Set up Redis caching
3. Implement query router

### Phase 3: Intelligence (Week 5-6)
1. Index data in Pinecone
2. Integrate LangChain
3. Fine-tune semantic search

### Phase 4: Production (Week 7-8)
1. Deploy on Kubernetes
2. Set up monitoring (Prometheus + Grafana)
3. Load testing & optimization

---

## 📊 Monitoring & Observability

```python
# monitoring/metrics.py

from prometheus_client import Counter, Histogram, Gauge
import time

query_duration = Histogram(
    'query_duration_seconds',
    'Time spent processing queries',
    ['query_type', 'status']
)

cache_hits = Counter(
    'cache_hits_total',
    'Number of cache hits',
    ['query_type']
)

active_queries = Gauge(
    'active_queries',
    'Number of currently executing queries'
)

def monitored_query(func):
    """Decorator for monitoring query performance"""
    def wrapper(*args, **kwargs):
        start = time.time()
        active_queries.inc()
        
        try:
            result = func(*args, **kwargs)
            duration = time.time() - start
            query_duration.labels(
                query_type=kwargs.get('type', 'unknown'),
                status='success'
            ).observe(duration)
            return result
        
        except Exception as e:
            query_duration.labels(
                query_type=kwargs.get('type', 'unknown'),
                status='error'
            ).observe(time.time() - start)
            raise
        
        finally:
            active_queries.dec()
    
    return wrapper
```

---

## 🎯 Key Takeaways

### ✅ Must-Have Components:
1. **BigQuery** - Serverless data warehouse
2. **Spark/Airflow** - Distributed processing
3. **Pinecone** - Vector search for intelligence
4. **Redis** - Response caching
5. **Kubernetes** - Auto-scaling infrastructure

### 🏆 Best Practices:
- **Partition** data by date
- **Pre-aggregate** common queries
- **Cache** heavily accessed results
- **Index** semantic insights
- **Monitor** everything

### 📈 Scalability Metrics:
- Handles **100GB+** data
- Supports **1000+** concurrent users
- Query response **< 2 seconds**
- **99.9%** uptime SLA
- Auto-scales **3-20 nodes**

---
