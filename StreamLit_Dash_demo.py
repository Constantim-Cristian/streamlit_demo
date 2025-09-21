import streamlit as st
import pandas as pd
import base64
import math
import uuid
from datetime import datetime
from pathlib import Path

# Core search & NLP dependencies
import json
import re
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import fuzz
import time

# Optional semantic search dependencies
try:
    from sentence_transformers import SentenceTransformer
    import faiss
    HAS_SEMANTIC = True
except ImportError:
    HAS_SEMANTIC = False


# -----------------------------
# Utility functions
# -----------------------------
def get_base64_of_bin_file(bin_file):
    """Convert binary file to base64 string (used for inline images/logos)."""
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()


def generate_product_uuid(store, title, price, quantity):
    """Generate stable UUID per product, based on store, title, price, and quantity."""
    unique_string = f"{store}|{title}|{price}|{quantity}"
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, unique_string))


# -----------------------------
# Text processing helpers
# -----------------------------
def normalize_text(text):
    """Lowercase, remove digits/symbols, collapse spaces for consistent matching."""
    text = text.lower()
    text = text.replace("-", " ").replace("_", " ")
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def enrich_text(text, synonyms):
    """Expand query/title using synonyms keys to improve recall during matching."""
    words = text.split()
    enriched = words.copy()
    for key, vals in synonyms.items():
        for val in vals:
            if val in words and key not in enriched:
                enriched.insert(0, key)
    return " ".join(enriched)


# -----------------------------
# Search system initialization
# -----------------------------
@st.cache_resource
def initialize_search_system():
    """
    Load today's product JSON and synonyms, build TF-IDF vectorizers and optionally
    initialize semantic model + FAISS index if dependency files are present.
    """
    with st.spinner("🔄 Initializing smart search system..."):
        try:
            script_dir = Path(__file__).parent
            today = datetime.now()
            date_str = today.strftime("%d-%m-%Y")
            products_file = rf"products_21-09-2025.json"

            # Load products JSON
            with open(products_file, 'r', encoding='utf-8') as f:
                products = json.load(f)

            # Load synonyms dictionary file (exec to populate synonyms dict)
            synonyms = {}
            synonyms_file = rf'dictonary.txt'
            with open(synonyms_file, 'r', encoding='utf-8') as f:
                content = f.read()
            local_vars = {'synonyms': synonyms}
            exec(content, {}, local_vars)
            synonyms = local_vars['synonyms']

            # Build normalized + enriched corpus from product titles
            corpus = []
            for p in products:
                normalized = normalize_text(p['Title'])
                enriched = enrich_text(normalized, synonyms)
                corpus.append(enriched)

            # TF-IDF models for unigram and bigram handling
            vectorizer_unigram = TfidfVectorizer(ngram_range=(1, 1))
            tfidf_matrix_unigram = vectorizer_unigram.fit_transform(corpus)

            vectorizer_bigram = TfidfVectorizer(ngram_range=(1, 2))
            tfidf_matrix_bigram = vectorizer_bigram.fit_transform(corpus)

            # Optionally load precomputed semantic embeddings + FAISS index
            semantic_model = None
            faiss_index = None
            products_vec = None

            if HAS_SEMANTIC:
                try:
                    vectors_file = r"vectors_gpu.pkl"
                    with open(vectors_file, "rb") as f:
                        data = pickle.load(f)
                        products_vec = data["products"]
                        embeddings = data["embeddings"]

                        # Normalize embeddings for inner-product based similarity
                        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

                        # Build FAISS index for fast top-k retrieval
                        d = embeddings.shape[1]
                        faiss_index = faiss.IndexFlatIP(d)
                        faiss_index.add(embeddings.astype('float32'))

                        semantic_model = SentenceTransformer("distiluse-base-multilingual-cased-v2", device="cpu")
                except Exception as e:
                    st.warning(f"⚠️ Semantic search files not found: {e}")

            return {
                'products': products,
                'synonyms': synonyms,
                'corpus': corpus,
                'vectorizer_unigram': vectorizer_unigram,
                'vectorizer_bigram': vectorizer_bigram,
                'tfidf_matrix_unigram': tfidf_matrix_unigram,
                'tfidf_matrix_bigram': tfidf_matrix_bigram,
                'semantic_model': semantic_model,
                'faiss_index': faiss_index,
                'products_vec': products_vec
            }

        except Exception as e:
            st.error(f"❌ Failed to load search system: {str(e)}")
            return None


# -----------------------------
# Search methods
# -----------------------------
def tfidf_search(query, search_data, fuzzy_threshold=60, candidate_limit=150):
    """TF-IDF search with a fuzzy-match bonus adjusted by RapidFuzz ratio."""
    if not search_data:
        return [], 0, "none"

    start_time = time.time()
    query_norm = normalize_text(query)
    is_single_word = len(query_norm.split()) == 1
    query_enriched = enrich_text(query_norm, search_data['synonyms'])

    # Single-word queries use unigram vectorizer, otherwise bigram vectorizer
    if is_single_word:
        q_vec = search_data['vectorizer_unigram'].transform([query_enriched])
        scores = cosine_similarity(q_vec, search_data['tfidf_matrix_unigram'])[0]
        method_detail = "unigram"
    else:
        q_vec = search_data['vectorizer_bigram'].transform([query_enriched])
        scores = cosine_similarity(q_vec, search_data['tfidf_matrix_bigram'])[0]
        method_detail = "bigram"

    # Efficient top candidate selection
    if len(scores) > candidate_limit:
        top_idx = np.argpartition(scores, -candidate_limit)[-candidate_limit:]
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
    else:
        top_idx = scores.argsort()[::-1]

    results = []
    for idx in top_idx:
        if scores[idx] < 0.05:
            break

        title = search_data['products'][idx]['Title']
        title_norm = normalize_text(title)
        title_enriched = enrich_text(title_norm, search_data['synonyms'])

        # Fuzzy matching ratio gives a bounded bonus or penalty to TF-IDF score
        ratio = fuzz.token_set_ratio(query_enriched, title_enriched)
        if ratio >= 95:
            fuzzy_bonus = 0.30
        elif ratio >= 85:
            fuzzy_bonus = 0.20 + (ratio - 85) * 0.008
        elif ratio >= 75:
            fuzzy_bonus = 0.10 + (ratio - 75) * 0.008
        elif ratio >= 60:
            fuzzy_bonus = 0.02 + (ratio - 60) * 0.004
        else:
            fuzzy_bonus = -0.40

        final_score = float(scores[idx]) * (1 + fuzzy_bonus)

        if ratio >= fuzzy_threshold:
            results.append({
                "Title": title,
                "Score": final_score,
                "Fuzzy": ratio,
                "Method": "tfidf",
                "Index": int(idx)
            })

    # Sort results by final hybrid score and fuzzy ratio
    results.sort(key=lambda x: (x['Score'], x['Fuzzy']), reverse=True)
    return results, time.time() - start_time, method_detail


def semantic_search_faiss(query, search_data, top_k=100):
    """Use semantic embeddings and FAISS to retrieve top-k semantically similar products."""
    if not search_data or not search_data['semantic_model'] or not search_data['faiss_index']:
        return [], 0

    start_time = time.time()
    query_vec = search_data['semantic_model'].encode([query], convert_to_numpy=True)
    query_vec = query_vec / np.linalg.norm(query_vec, axis=1, keepdims=True)

    actual_k = min(top_k, len(search_data['products_vec']))
    sims, idxs = search_data['faiss_index'].search(query_vec.astype('float32'), actual_k)

    results = []
    for i, idx in enumerate(idxs[0]):
        if idx < len(search_data['products_vec']):
            product = search_data['products_vec'][idx]
            results.append({
                "Title": product["Title"],
                "Category": f"{product.get('categoryname', '')} - {product.get('usecategory', '')}",
                "Similarity": float(sims[0][i]),
                "Method": "semantic",
                "Index": int(idx)
            })

    return results, time.time() - start_time


def smart_search(query, search_data, fuzzy_threshold=60, min_results=6, min_tfidf_score=0.55):
    """
    Hybrid orchestration: run TF-IDF first, fall back to semantic search if TF-IDF
    yields too few or too weak results.
    """
    if not search_data:
        return [], {'method': 'none', 'tfidf_time': 0, 'semantic_time': 0,
                    'total_time': 0, 'results_count': 0}

    start_time = time.time()
    results, tfidf_time, method_detail = tfidf_search(query, search_data, fuzzy_threshold, 150)

    too_few = len(results) < min_results
    too_weak = not results or results[0]["Score"] < min_tfidf_score

    method_used = f"tfidf ({method_detail})"
    semantic_time = 0
    if (too_few or too_weak) and search_data.get('semantic_model'):
        results, semantic_time = semantic_search_faiss(query, search_data, 100)
        method_used = "semantic"

    total_time = time.time() - start_time
    return results, {
        'method': method_used,
        'tfidf_time': tfidf_time,
        'semantic_time': semantic_time,
        'total_time': total_time,
        'results_count': len(results)
    }


# -----------------------------
# Search helpers
# -----------------------------
def filter_dataframe_by_search(df, search_results):
    """Return rows for titles present in the provided search results."""
    if not search_results:
        return df
    search_titles = set(result['Title'] for result in search_results)
    return df[df['Title'].isin(search_titles)]


def get_search_score_for_product(title, search_results):
    """Return the raw score associated with a product title in search results."""
    if not search_results:
        return 0
    for result in search_results:
        if result['Title'] == title:
            if result['Method'] == 'tfidf':
                return result['Score']
            else:
                return result['Similarity'] * 100
    return 0


def get_search_score_for_display(title, search_results):
    """Return a score scaled for UI display (0-100)."""
    if not search_results:
        return 0
    for result in search_results:
        if result['Title'] == title:
            if result['Method'] == 'tfidf':
                return result['Score']
            else:
                return result['Similarity'] * 100
    return 0


# -----------------------------
# Streamlit UI setup & data load
# -----------------------------
st.set_page_config(layout="wide")


def local_css(file_name):
    """Inject local CSS if file exists (gracefully ignore if missing)."""
    try:
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        pass


local_css(rf"STCSS.css")

# Build or load search data/models
search_data = initialize_search_system()
today = datetime.now()
date_str = today.strftime("%d-%m-%Y")


@st.cache_data
def load_product_data():
    """Load today's scraped products JSON into a DataFrame and create UUIDs."""
    script_dir = Path(__file__).parent
    filename = rf"products_21-09-2025.json"

    df = pd.read_json(filename)
    df = df.sort_values('MetrPrice', ascending=True)
    df['product_uuid'] = df.apply(
        lambda row: generate_product_uuid(row['Store'], row['Title'],
                                          row['Current Price'], row['Quantity']),
        axis=1
    )
    return df


@st.cache_data
def create_product_lookup(df):
    """Create a lookup dict keyed by product_uuid for quick access in the UI/cart."""
    lookup = {}
    for _, row in df.iterrows():
        lookup[row['product_uuid']] = {
            'store': row['Store'],
            'title': row['Title'],
            'price': row['Current Price'],
            'quantity': row['Quantity'],
            'unit': row['Unit'],
            'image_url': row['Image URL'],
            'metr_price': row['MetrPrice'],
            'prod_link': row['Product Link']
        }
    return lookup


df = load_product_data()
product_lookup = create_product_lookup(df)
ORIGINAL_STORE_ORDER = list(df['Store'].unique())

script_dir = Path(__file__).parent
parent_dir = script_dir.parent
icon_dir = r'icon'

STORE_LOGOS = {
    'auchan-hypermarket-titan': rf"{icon_dir}/Auchan_2018.svg",
    'carrefour-hypermarket-mega-mall': rf"{icon_dir}/Carrefour_2009_(Horizontal).svg",
    'freshful-now': rf"{icon_dir}/Freshful-logo.svg",
    'kaufland-pantelimon': rf"{icon_dir}/Kaufland_1984_wordmark.svg",
    'Penny': rf"{icon_dir}/Penny_Markt_2012.svg",
    'profi-baia-de-arama': rf"{icon_dir}/Profi_2016_no_symbol.svg"
}


@st.cache_data
def get_store_logo(store_name):
    """Resolve the best matching logo file path for a store (returns None if not found)."""
    logo_path = STORE_LOGOS.get(store_name)
    if logo_path:
        logo_path = Path(logo_path)
        if logo_path.exists():
            return str(logo_path)

    icon_path = Path(icon_dir)
    if icon_path.exists():
        for file in icon_path.iterdir():
            if store_name.lower().replace(' ', '-') in file.name.lower():
                return str(file)
    return None


# -----------------------------
# Session state initialization
# -----------------------------
if 'selected_products' not in st.session_state:
    st.session_state.selected_products = {store: set() for store in df['Store'].unique()}

if 'product_quantities' not in st.session_state:
    st.session_state.product_quantities = {}

if 'units_multiselect' not in st.session_state:
    st.session_state.units_multiselect = []

if 'categories_multiselect' not in st.session_state:
    st.session_state.categories_multiselect = []

if 'widget_reset_counter' not in st.session_state:
    st.session_state.widget_reset_counter = 0

if 'search_results' not in st.session_state:
    st.session_state.search_results = []

if 'search_timing' not in st.session_state:
    st.session_state.search_timing = {}

min_price = math.floor(df['Current Price'].min())
max_price = math.ceil(df['Current Price'].max())


def clear_all_selections():
    """Reset selections, quantities and UI reset counter to force widget keys refresh."""
    st.session_state.selected_products = {store: set() for store in df['Store'].unique()}
    st.session_state.product_quantities = {}
    st.session_state.units_multiselect = []
    st.session_state.categories_multiselect = []
    st.session_state.widget_reset_counter += 1


def remove_product_from_selection(store, product_uuid):
    """Remove UUID from selected set and delete any stored quantity for it."""
    st.session_state.selected_products[store].discard(product_uuid)
    if product_uuid in st.session_state.product_quantities:
        del st.session_state.product_quantities[product_uuid]
    st.session_state.widget_reset_counter += 1


# -----------------------------
# Sidebar: Search & Filters
# -----------------------------
with st.sidebar:
    st.header("🔍 Smart Search & Filters")

    if not HAS_SEMANTIC:
        st.warning("⚠️ Semantic search unavailable. Install: pip install sentence-transformers faiss-cpu")

    st.caption("Uses AI-powered search with fuzzy matching and semantic understanding.")

    # Smart search text input
    search_query = st.text_input(
        "Search for products:",
        placeholder="e.g., ardei capia, lapte, oua...",
        help="Try natural language queries like 'red peppers' or 'milk products'"
    )

    # If query changed, run smart_search and store results + timing in session_state
    if search_query and search_data:
        if search_query != st.session_state.get('last_query', ''):
            with st.spinner("🔍 Searching..."):
                results, timing = smart_search(search_query, search_data)
                st.session_state.search_results = results
                st.session_state.search_timing = timing
                st.session_state.last_query = search_query

        # Display quick timing/method info to the user
        if st.session_state.search_results:
            timing = st.session_state.search_timing
            st.markdown(f"""
                <div class="search-timing">
                ✅ Found {timing['results_count']} products in {timing['total_time']:.3f}s<br>
                Method: {timing['method'].upper()}
                </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("No products found. Try different search terms.")
    elif search_query and not search_data:
        st.error("Search system not available. Using basic text search.")

    st.markdown("<br>", unsafe_allow_html=True)

    # Start with a copy of df, then narrow by search + other filters
    filtered_df = df.copy()
    if search_query:
        if st.session_state.search_results:
            filtered_df = filter_dataframe_by_search(df, st.session_state.search_results)
        else:
            filtered_df = filtered_df[filtered_df['Title'].str.contains(search_query, case=False, na=False)]

    # Dynamic unit filter based on currently visible products
    available_units = sorted(filtered_df['Unit'].dropna().unique().tolist())
    selected_units = st.multiselect("📦 Filter by Unit",
                                    options=available_units,
                                    key="units_multiselect",
                                    help="Select one or more units to filter products")

    # Dynamic category filter based on currently visible products
    available_categories = sorted(filtered_df['usecategory'].dropna().unique().tolist())
    selected_categories = st.multiselect("📂 Filter by Category",
                                         options=available_categories,
                                         key="categories_multiselect",
                                         help="Select one or more categories to filter products")

    st.markdown("---")

    price_range = st.slider("Price range (LEI):", min_value=min_price, max_value=max_price,
                            value=(min_price, max_price), step=1)

    st.button("🗑️ Clear All Selections", type="secondary", on_click=clear_all_selections)

    # Show active filters summary
    active_filters_count = 0
    if selected_units:
        active_filters_count += len(selected_units)
    if selected_categories:
        active_filters_count += len(selected_categories)

    if active_filters_count > 0:
        st.markdown(f"**Active Filters: {active_filters_count}**")
        if selected_units:
            st.markdown(f"**Units:** {', '.join(selected_units[:3])}{'...' if len(selected_units) > 3 else ''}")
        if selected_categories:
            st.markdown(f"**Categories:** {', '.join(selected_categories[:2])}{'...' if len(selected_categories) > 2 else ''}")

# -----------------------------
# Apply additional filters to the main dataframe
# -----------------------------
if selected_units:
    filtered_df = filtered_df[filtered_df['Unit'].isin(selected_units)]

if selected_categories:
    filtered_df = filtered_df[filtered_df['usecategory'].isin(selected_categories)]

filtered_df = filtered_df[
    (filtered_df['Current Price'] >= price_range[0]) &
    (filtered_df['Current Price'] <= price_range[1])
]

# -----------------------------
# Apply smart scoring + sorting when search active
# -----------------------------
if search_query and st.session_state.search_results:
    filtered_df['search_score'] = filtered_df['Title'].apply(
        lambda title: get_search_score_for_product(title, st.session_state.search_results)
    )

    # Build a 4-group priority sort to boost relevancy visually
    query_words = normalize_text(search_query).split()
    first_query_word = query_words[0] if query_words else ""

    def categorize_result(row):
        timing = st.session_state.get('search_timing', {})
        is_semantic = timing.get('method', '').startswith('semantic')

        if is_semantic:
            # Semantic results: two-group split by a high threshold for simplicity
            if row['search_score'] > 75:
                return 3
            else:
                return 4
        else:
            # TF-IDF: check for positional matches (stronger signal)
            title_norm = normalize_text(row['Title'])
            title_words = title_norm.split()
            if not title_words:
                return 4

            # Exact sequence match at start -> highest priority
            if len(query_words) > 1 and len(title_words) >= len(query_words):
                if title_words[:len(query_words)] == query_words:
                    return 1

            # First-word match -> second priority
            if title_words[0] == first_query_word:
                return 2

            # High TF-IDF score -> third priority
            if row['search_score'] > 0.59:
                return 3

            # Everything else -> lowest priority
            return 4

    filtered_df['sort_group'] = filtered_df.apply(categorize_result, axis=1)
    filtered_df = filtered_df.sort_values(['sort_group', 'MetrPrice'], ascending=[True, True]).reset_index(drop=True)
    st.markdown("**Smart Sort:** 4-tier grouping by search relevance, each sorted by best value")
else:
    filtered_df = filtered_df.sort_values('MetrPrice', ascending=True)
    st.markdown("**Sort by Value Per Quantity**")

# -----------------------------
# Build store-wise views and product cards
# -----------------------------
filtered_stores = set(filtered_df['Store'].unique())
stores = [store for store in ORIGINAL_STORE_ORDER if store in filtered_stores]

store_dfs = {store: filtered_df[filtered_df['Store'] == store].reset_index(drop=True) for store in stores}
rows_per_page = 5
rows = [st.columns(3), st.columns(3)]

for idx, store in enumerate(stores):
    col = rows[idx // 3][idx % 3]
    with col:
        st.markdown("<div style='padding-top: 20px;'></div>", unsafe_allow_html=True)
        logo_path = get_store_logo(store)
        if logo_path:
            try:
                st.markdown(
                    f"""
                    <div style="display: flex; justify-content: center; align-items: center;">
                        <img src="data:image/svg+xml;base64,{get_base64_of_bin_file(logo_path)}" style="width: 200px; height: 80px; border-radius: 8px;" />
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            except:
                st.image(logo_path, width=50)
        else:
            st.markdown(f"<h3 style='text-align: center;'>{store}</h3>", unsafe_allow_html=True)

        st.markdown("<div style='padding-bottom: 15px;'></div>", unsafe_allow_html=True)
        store_df = store_dfs[store].copy()
        total_pages = len(store_df) // rows_per_page + (len(store_df) % rows_per_page > 0)
        page_key = f'page_{store}'

        with st.container(height=300):
            current_page = st.session_state.get(page_key, 1)
            if current_page > total_pages:
                current_page = 1
                st.session_state[page_key] = 1

            page = current_page
            start_idx = (page - 1) * rows_per_page
            end_idx = start_idx + rows_per_page
            paginated_df = store_df.iloc[start_idx:end_idx].copy()

            for _, row in paginated_df.iterrows():
                image_url = row['Image URL']
                product = row['Title']
                product_uuid = row['product_uuid']
                price = row['Current Price']
                quantity = row['Quantity']
                unit = row["Unit"]
                prodlink = row["Product Link"]

                is_selected = product_uuid in st.session_state.selected_products[store]
                cols = st.columns([3, 2], gap="small")

                st.markdown("<div style='padding-top: 10px;'>", unsafe_allow_html=True)

                with cols[0]:
                    st.markdown(
                        f"""
                        <div class="image-wrapper">
                                <a href={prodlink}>
                                    <img src="{image_url}" style="width: 300px; height: 160px; object-fit: cover; border-radius: 8px;" />
                                </a>
                            <a href="{image_url}" target="_blank" class="fullscreen-icon">🔍</a>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                with cols[1]:
                    checkbox_key = f"main_{store}_{product_uuid}_{st.session_state.widget_reset_counter}"

                    selected = st.checkbox("Select", value=is_selected, key=checkbox_key)

                    if selected != is_selected:
                        if selected:
                            st.session_state.selected_products[store].add(product_uuid)
                            if product_uuid not in st.session_state.product_quantities:
                                st.session_state.product_quantities[product_uuid] = 1
                        else:
                            st.session_state.selected_products[store].discard(product_uuid)
                            if product_uuid in st.session_state.product_quantities:
                                del st.session_state.product_quantities[product_uuid]

                    short_title = product[:19] + "..." if len(product) > 19 else product
                    search_score = get_search_score_for_display(product, st.session_state.search_results) if st.session_state.get('search_results') else 0
                    st.markdown(
                    f"""
                        <div class="product-title-container">
                            <span class="product-title-short">{short_title}</span>
                            <div class="tooltip">{product}</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    st.markdown(f"LEI {round(price,2)} / {quantity} {unit}")

        if total_pages > 1:
            current_page = st.session_state.get(page_key, 1)
            if current_page > total_pages:
                st.session_state[page_key] = total_pages
                current_page = total_pages

            st.number_input("Page", min_value=1, max_value=max(1, total_pages),
                            value=current_page, key=page_key, step=1,
                            help=f"Page navigation for {store} ({total_pages} pages total)",
                            label_visibility="collapsed")

# -----------------------------
# Selected products (cart) display and controls
# -----------------------------
st.subheader("🛒 Selected Products by Store")

if len(stores) == 0:
    st.info("No products found matching your search and filters. Try adjusting your search terms or filters.")
else:
    with st.container(height=600):
        store_columns = st.columns(len(stores))
        grand_total = 0

        for idx, store in enumerate(stores):
            with store_columns[idx]:
                selected_product_uuids = st.session_state.selected_products[store]

                store_total = 0
                for uuid in selected_product_uuids:
                    if uuid in product_lookup:
                        product_price = product_lookup[uuid]['price']
                        quantity = st.session_state.product_quantities.get(uuid, 1)
                        store_total += product_price * quantity

                grand_total += store_total

                logo_path = get_store_logo(store)
                if logo_path:
                    try:
                        st.markdown(
                            f"""
                            <div style="display: flex; justify-content: left; align-items: left;">
                                <img src="data:image/svg+xml;base64,{get_base64_of_bin_file(logo_path)}" style="width: 90px; height: 35px; border-radius: 8px; padding-bottom: 5px;" />
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    except:
                        st.markdown(f"<h4 style='text-align: center;'>{store}</h4>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<h4 style='text-align: center;'>{store}</h4>", unsafe_allow_html=True)

                st.markdown(f'<div class="total-price">{store_total:.2f} LEI</div>', unsafe_allow_html=True)

                if selected_product_uuids:
                    for product_uuid in selected_product_uuids.copy():
                        if product_uuid in product_lookup:
                            product_info = product_lookup[product_uuid]
                            product = product_info['title']
                            price = product_info['price']
                            quantity = product_info['quantity']
                            unit = product_info['unit']
                            current_product_image = product_info['image_url']
                            prodlink = product_info["prod_link"]
                            metprice = product_info["metr_price"]

                            current_qty = st.session_state.product_quantities.get(product_uuid, 1)

                            cols = st.columns([3, 1], gap="small")
                            with cols[0]:
                                st.markdown(
                                    f"""
                                    <div class="image-wrapper">
                                            <a href={prodlink}>
                                                <img src="{current_product_image}" style="width: 200px; height: 100px; object-fit: cover; border-radius: 8px; padding-top: 0px; padding-bottom: 5px;" />
                                            </a>
                                        <a href="{current_product_image}" target="_blank" class="fullscreen-icon">🔍</a>
                                    </div>
                                    """,
                                    unsafe_allow_html=True
                                )

                                short_title = product[:20] + "..." if len(product) > 20 else product
                                st.markdown(
                                    f"""
                                    <div class="product-title-container">
                                        <span class="product-title-short">{short_title}</span>
                                        <div class="tooltip">{product}</div>
                                    </div>
                                    """,
                                    unsafe_allow_html=True
                                )
                                total_product_price = price * current_qty
                                total_quant = quantity * current_qty
                                st.markdown(f"**Total:** \n\n **LEI {total_product_price:.2f}** \n\n **{total_quant} {unit}**")

                            with cols[1]:
                                if st.button("➖", key=f"dec_{store}_{product_uuid}"):
                                    if current_qty > 1:
                                        st.session_state.product_quantities[product_uuid] = current_qty - 1
                                        st.rerun()

                                st.markdown(f'<div class="qty-indic" style=" padding-left: 19px; " >{current_qty}</div>', unsafe_allow_html=True)

                                st.write(f"")

                                if st.button("➕", key=f"inc_{store}_{product_uuid}"):
                                    st.session_state.product_quantities[product_uuid] = current_qty + 1
                                    st.rerun()

                                if st.button("🗑️", key=f"remove_{store}_{product_uuid}"):
                                    remove_product_from_selection(store, product_uuid)
                                    st.rerun()

                            st.markdown("---")
                else:
                    st.write("No products selected")
