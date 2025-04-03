import pandas as pd
import io
from minio import Minio

# Configure MinIO client
minio_client = Minio(
    "minio:9000",
    access_key="minioadmin",
    secret_key="minioadmin",
    secure=False
)

# Load a view from the access-zone
def load_view(view_name):
    """Load a pre-generated view from the access zone"""
    object_path = f"unstructured/{view_name}.parquet"

    try:
        # Get object from MinIO
        response = minio_client.get_object("access-zone", object_path)

        # Read into pandas
        df = pd.read_parquet(io.BytesIO(response.read()))

        print(f"Loaded {view_name} with {len(df)} records")
        return df

    except Exception as e:
        print(f"Error loading view: {e}")
        return None

# Examples of loading different views
customer_feedback_df = load_view("customer_feedback")
product_info_df = load_view("product_information")
market_analysis_df = load_view("market_analysis")

# Now you can use standard pandas operations on this data
if customer_feedback_df is not None:
    # Example: Filter by score
    high_relevance = customer_feedback_df[customer_feedback_df['score'] > 0.8]
    print(f"High relevance feedback: {len(high_relevance)} records")

    # Example: Group by source
    source_counts = customer_feedback_df.groupby('source').size()
    print("Feedback by source:")
    print(source_counts)