import streamlit as st
import re
import pandas as pd
from spell_engine import SpellCheckerEngine

# 1. Page Configuration & Setup
st.set_page_config(page_title="Corpus Spell Checker", layout="wide", initial_sidebar_state="collapsed")
st.markdown("""
    <style>
        /* Target the main container that holds everything */
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            padding-left: 2rem;   
            padding-right: 2rem;  
            max-width: 98%;       
        }

        /* 1. Reduce the font size of the text inside the checkboxes */
        /* Streamlit now wraps checkbox text in a <p> tag */
        div[data-testid="stCheckbox"] p {
            font-size: 0.85rem !important;
            line-height: 1.2 !important;
        }
        
        /* 2. Reduce the vertical gap between items inside the expander */
        /* Streamlit uses an internal flexbox called stVerticalBlock inside expanders */
        [data-testid="stExpanderDetails"] [data-testid="stVerticalBlock"] {
            gap: 0.1rem !important; 
        }

        [data-testid="stExpanderDetails"] {
            padding-top: 0.25rem !important;
            padding-bottom: 0.25rem !important;
        }

        /* 3. Compress the height and padding of the checkbox containers */
        div[data-testid="stCheckbox"] {
            min-height: 1.5rem !important;
            padding-bottom: 0rem !important;
            margin-bottom: 0rem !important;
        }
        
        /* Compress the internal label padding of the checkbox */
        div[data-testid="stCheckbox"] label {
            padding-bottom: 0.1rem !important;
            min-height: auto !important;
        }
        
        /* 4. Reduce the font size of the expander titles */
        /* The old .streamlit-expanderHeader class was deprecated */
        [data-testid="stExpander"] summary p {
            font-size: 0.95rem !important;
        }
        [data-testid="stExpanderDetails"] [data-testid="stAlert"] {
            padding-top: 0.01rem !important;
            padding-bottom: 0.3rem !important;
            padding-left: 0.8rem !important;
            padding-right: 0.8rem !important;
            min-height: auto !important;
            margin-bottom: 0rem !important;
        }

        /* Reduce the text size inside those alert boxes */
        [data-testid="stExpanderDetails"] [data-testid="stAlert"] p {
            font-size: 0.85rem !important;
            margin-top: -10px !important;
            margin-bottom: 0px !important;
        }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_engine():
    return SpellCheckerEngine('central_bank_corpus.txt')

with st.spinner('Loading NLP Models and Central Bank Corpus...'):
    engine = load_engine()

# Initialize session state variables
if 'text_input' not in st.session_state:
    st.session_state.text_input = "The central bank raised the interest rats because of the graffe. It navigates well." #default text
if 'errors' not in st.session_state:
    st.session_state.errors = []
if 'check_run' not in st.session_state:
    st.session_state.check_run = False

# Callback Functions
def apply_correction(original, suggestion):
    """Replaces the word and instantly re-runs the spell check"""
    if original[0].isupper():
        suggestion = suggestion.capitalize()
    new_text = re.sub(
        rf"\b{re.escape(original)}\b",
        suggestion,
        st.session_state.text_input,
        count=1,
        flags=re.IGNORECASE
    )
    st.session_state.text_input = new_text
    st.session_state.errors = engine.check_sentence(new_text)

def clear_workspace():
    """Cleans the text box and resets the app state"""
    st.session_state.text_input = ""
    st.session_state.errors = []
    st.session_state.check_run = False

def run_spellcheck():
    """Runs the engine on the current text"""
    # Wrap the heavy math in a spinner to give the user visual feedback
    with st.spinner('Analyzing text and calculating probabilities...'):
        st.session_state.errors = engine.check_sentence(st.session_state.text_input)
        st.session_state.check_run = True

# 2. Main Layout (Two Columns)
col1, space, col2 = st.columns([1.5, 0.02, 1.0])

# LEFT PANEL: The Workspace
with col1:
    st.subheader("📝 Central Banks Speech Corpus")
    
    # Using key="text_input" forces Streamlit to clear the box properly
    st.text_area(
        "Text Editor (Max 500 characters)", 
        key="text_input", 
        max_chars=500, 
        height=200
    )
    
    # Action Buttons (Using Callbacks)
    c1, c2 = st.columns(2)
    with c1:
        st.button("Check Spelling", use_container_width=True, type="primary", on_click=run_spellcheck)
    with c2:
        st.button("Clear Text", use_container_width=True, on_click=clear_workspace)
    # Always Visible Color Legend
    st.markdown("""
    <div style="margin-bottom: 10px; font-size: 14px;">
        <strong>Legend:</strong>&nbsp;&nbsp;
        <span style="background-color: #ff4b4b; color: white; padding: 2px 6px; border-radius: 4px;">Typo (Non-Word)</span>&nbsp;&nbsp;
        <span style="background-color: #ffa500; color: white; padding: 2px 6px; border-radius: 4px;">Context (Real-Word)</span>&nbsp;&nbsp;
        <span style="background-color: #1e90ff; color: white; padding: 2px 6px; border-radius: 4px;">Out-of-Domain (OOV)</span>
    </div>
    """, unsafe_allow_html=True)
    # Document Review
    if st.session_state.check_run:
        st.markdown("### Document Review") 
        # Prompt when no mistakes are found
        if not st.session_state.errors:
            st.success("✅ No spelling or context errors found in this text!")
            st.markdown(f'<div style="border:1px solid #ddd; padding:15px; border-radius:5px; background-color:#262730;">{st.session_state.text_input}</div>', unsafe_allow_html=True)
            
        else:
            highlighted_text = st.session_state.text_input
            
            # Sort errors in REVERSE order by their start position. This prevents HTML tags from messing up the indices of earlier words
            sorted_errors = sorted(st.session_state.errors, key=lambda x: x['start'], reverse=True)
            
            for err in sorted_errors:
                start = err['start']
                end = err['end']
                word = err['original']
                
                if "Non-Word" in err['type']:
                    color = "#ff4b4b"
                elif "Context" in err['type']:
                    color = "#ffa500"
                else:
                    color = "#1e90ff" # OOV
                
                # Create the HTML span
                html_tag = f'<span style="background-color: {color}; color: white; padding: 2px 4px; border-radius: 4px; font-weight: bold;">{word}</span>'
                
                # Slice the string and insert the tag
                highlighted_text = highlighted_text[:start] + html_tag + highlighted_text[end:]
            
            st.markdown(f'<div style="border:1px solid #ddd; padding:15px; border-radius:5px; background-color:#262730;">{highlighted_text}</div>', unsafe_allow_html=True)

# RIGHT PANEL: Tools & Feedback along with Corpus dictionary
with col2:
    # st.subheader("🛠️ Tools & Feedback")
    
    st.subheader("**Correction Suggestions**")
    st.markdown("<hr style='margin-top: 0px; margin-bottom: 10px;'/>", unsafe_allow_html=True)
    
    if not st.session_state.check_run:
        st.info("Run 'Check Spelling' to see suggestions here.")
    elif not st.session_state.errors:
        st.success("No suggestions needed.")
    else:
        # Create an expanding box for every error found
        for err in st.session_state.errors:
            if "Non-Word" in err['type']:
                icon = "🚨"
            elif "Context" in err['type']:
                icon = "⚠️"
            else:
                icon = "ℹ️" # OOV
            
            with st.expander(f"{icon} '{err['original']}' ({err['type']})", expanded=True):
                if not err['suggestions']:
                    st.info("Valid English word, but not found in the Central Bank corpus.")
                else:
                    for sugg, dist, prob in err['suggestions']:
                        btn_key = f"btn_{err['index']}_{err['original']}_{sugg}"
                        button_label = f"✅ {sugg} (Dist: {dist}   || Prob: {prob*100:.4f}%)"
                        
                        # Use the callback! Streamlit passes 'args' into the re-run function automatically
                        st.button(
                            button_label, 
                            key=btn_key, 
                            on_click=apply_correction, 
                            args=(err['original'], sugg)
                        )
                    
    st.markdown("", unsafe_allow_html=True)
    
    # Dictionary Section (Two Tabs)
    st.subheader("**Corpus Dictionary**")
    st.markdown("<hr style='margin-top: 0px; margin-bottom: 10px;'/>", unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["🔤 Unigram Dictionary", "🔗 Bigram Dictionary"])

    # --- TAB 1: UNIGRAM SEARCH ---
    with tab1:
        search_uni = st.text_input("🔍 Search single words...", "", key="uni_search").lower().strip()
        
        unigram_data = []
        
        # Iterate through the vocabulary to get counts and probabilities
        for word in engine.vocab:
            if not search_uni or search_uni in word:
                unigram_data.append({
                    "Word": word,
                    "Count": engine.unigram_counts[word],
                    "Probability": f"{engine.get_word_probability(word) * 100:.6f}%" 
                })
        
        if unigram_data:
            # Sort highest frequency to the top
            unigram_data = sorted(unigram_data, key=lambda x: x["Count"], reverse=True)
            df_unigrams = pd.DataFrame(unigram_data)
            
            st.caption(f"**Vocabulary Stats:** Total words: {engine.total_words:,} | Unique: {len(engine.vocab):,}")
            
            st.dataframe(
                df_unigrams, 
                height=300, 
                use_container_width=True, 
                hide_index=True
            )
        else:
            st.warning(f"No words found containing '{search_uni}'.")

    # --- TAB 2: BIGRAM SEARCH ---
    with tab2:
        search_bi = st.text_input("🔍 Search word pairings...", placeholder="e.g., policy, market", key="bi_search").lower().strip()
        
        if search_bi:
            matching_bigrams = []
            
            for (w1, w2), count in engine.bigram_counts.items():
                if w1 == search_bi or w2 == search_bi:
                    prob = engine.get_bigram_probability(w1, w2)
                    matching_bigrams.append({
                        "Word 1": w1,
                        "Word 2": w2,
                        "Count": count,
                        "Probability": f"{prob * 100:.3f}%"
                    })
            
            if matching_bigrams:
                # Sort highest frequency to the top
                matching_bigrams = sorted(matching_bigrams, key=lambda x: x["Count"], reverse=True)
                
                # Convert to DataFrame for a clean table render
                df_bigrams = pd.DataFrame(matching_bigrams)
                
                styled_df = df_bigrams.style.set_properties(
                    subset=['Count', 'Probability'], 
                    **{'text-align': 'center'}
                )
                
                # Render the styled dataframe
                st.dataframe(
                    styled_df, height=300, use_container_width=True, hide_index=True)
            else:
                st.warning(f"No bigrams found containing '{search_bi}'.")
        else:
            st.info("Enter a word to see its contextual pairings.")