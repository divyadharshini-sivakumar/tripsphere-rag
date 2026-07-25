import streamlit as st

from hybrid_search import TechMartHybridSearch


# =========================================================
# 1. Page configuration
# =========================================================
st.set_page_config(
    page_title="TechMart Hybrid Search",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# 2. Styling
# =========================================================
st.markdown(
    """
    <style>
        .main-title {
            font-size: 2.7rem;
            font-weight: 800;
            margin-bottom: 0.2rem;
        }

        .subtitle {
            font-size: 1.05rem;
            color: #AAAAAA;
            margin-bottom: 1.5rem;
        }

        .product-card {
            border: 1px solid rgba(255, 255, 255, 0.12);
            border-radius: 12px;
            padding: 1.1rem;
            margin-bottom: 1rem;
            background: rgba(255, 255, 255, 0.03);
        }

        .product-name {
            font-size: 1.25rem;
            font-weight: 700;
        }

        .score {
            font-weight: 700;
        }
    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# 3. Load hybrid search engine
# =========================================================
@st.cache_resource(show_spinner=False)
def load_engine():
    return TechMartHybridSearch()


try:
    with st.spinner(
        "Loading TechMart products and creating the vector database..."
    ):
        engine = load_engine()

except Exception as error:
    st.error(
        f"Failed to initialize TechMart Hybrid Search: {error}"
    )
    st.stop()


# =========================================================
# 4. Header
# =========================================================
st.markdown(
    '<div class="main-title">'
    '🛒 TechMart Hybrid Search System'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Search TechMart products using vector search, '
    'BM25 keyword search and metadata filters.'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# 5. Sidebar filters
# =========================================================
with st.sidebar:
    st.header("🔍 Search Filters")

    categories = [
        "All",
        *engine.get_categories()
    ]

    brands = [
        "All",
        *engine.get_brands()
    ]

    selected_category = st.selectbox(
        "Category",
        categories
    )

    selected_brand = st.selectbox(
        "Brand",
        brands
    )

    price_column1, price_column2 = st.columns(2)

    with price_column1:
        minimum_price = st.number_input(
            "Min price",
            min_value=0.0,
            value=0.0,
            step=10.0
        )

    with price_column2:
        maximum_price = st.number_input(
            "Max price",
            min_value=0.0,
            value=5000.0,
            step=10.0
        )

    minimum_rating = st.slider(
        "Minimum rating",
        min_value=0.0,
        max_value=5.0,
        value=0.0,
        step=0.5
    )

    top_k = st.slider(
        "Number of results",
        min_value=1,
        max_value=20,
        value=5
    )

    st.markdown("---")
    st.subheader("⚖️ Search Weights")

    vector_weight = st.slider(
        "Vector weight",
        min_value=0.0,
        max_value=1.0,
        value=0.6,
        step=0.1
    )

    keyword_weight = 1.0 - vector_weight

    st.write(
        f"Keyword weight: **{keyword_weight:.1f}**"
    )

    st.markdown("---")
    st.subheader("📊 Database Information")

    st.metric(
        "Products",
        engine.get_product_count()
    )

    st.metric(
        "Indexed vectors",
        engine.get_vector_count()
    )

    st.caption(
        "ChromaDB is created in memory for Streamlit Cloud."
    )


# =========================================================
# 6. Search form
# =========================================================
with st.form("techmart_search_form"):
    search_query = st.text_input(
        "What product are you looking for?",
        placeholder=(
            "Example: affordable wireless headphones "
            "with good battery life"
        )
    )

    search_button = st.form_submit_button(
        "Search TechMart 🚀",
        use_container_width=True
    )


# =========================================================
# 7. Search results
# =========================================================
if search_button:
    if not search_query.strip():
        st.warning(
            "Please enter a product search query."
        )

    elif maximum_price < minimum_price:
        st.warning(
            "Maximum price must be greater than minimum price."
        )

    else:
        with st.spinner(
            "Running vector search, keyword search and filtering..."
        ):
            try:
                results = engine.search(
                    query=search_query,
                    category=selected_category,
                    brand=selected_brand,
                    min_price=minimum_price,
                    max_price=maximum_price,
                    min_rating=minimum_rating,
                    top_k=top_k,
                    vector_weight=vector_weight,
                    keyword_weight=keyword_weight
                )

                if not results:
                    st.warning(
                        "No products matched your query and filters."
                    )

                else:
                    st.success(
                        f"Found {len(results)} matching products."
                    )

                    for rank, product in enumerate(
                        results,
                        start=1
                    ):
                        st.markdown(
                            f"""
                            <div class="product-card">
                                <div class="product-name">
                                    {rank}. {product['name']}
                                </div>

                                <p>
                                    <b>Product ID:</b>
                                    {product['product_id']}
                                </p>

                                <p>
                                    <b>Category:</b>
                                    {product['category']}
                                    &nbsp; | &nbsp;
                                    <b>Brand:</b>
                                    {product['brand']}
                                </p>

                                <p>
                                    <b>Price:</b>
                                    ${product['price']:.2f}
                                    &nbsp; | &nbsp;
                                    <b>Rating:</b>
                                    {product['rating']} / 5
                                    &nbsp; | &nbsp;
                                    <b>Stock:</b>
                                    {product['stock']}
                                </p>

                                <p>
                                    {product['description']}
                                </p>

                                <p class="score">
                                    Hybrid Score:
                                    {product['hybrid_score']:.4f}
                                    &nbsp; | &nbsp;
                                    Vector Score:
                                    {product['vector_score']:.4f}
                                    &nbsp; | &nbsp;
                                    Keyword Score:
                                    {product['keyword_score']:.4f}
                                </p>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

            except Exception as error:
                st.error(
                    f"Search failed: {error}"
                )


# =========================================================
# 8. Initial instructions
# =========================================================
else:
    st.info(
        "Enter a product requirement, select optional metadata "
        "filters and click **Search TechMart**."
    )

    st.markdown(
        """
        ### Sample searches

        - Wireless headphones with long battery life
        - Affordable laptop for students
        - Gaming mouse with high accuracy
        - Smartwatch for fitness tracking
        - Lightweight office laptop
        """
    )