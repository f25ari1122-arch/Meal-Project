import pickle
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="SmartBite", page_icon="🍽️", layout="wide")

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
CATEGORIES = ["Breakfast", "Lunch", "Dinner"]


@st.cache_resource
def load_models():
    with open("price_model.pkl", "rb") as f:
        price_model = pickle.load(f)
    with open("meal_recommender.pkl", "rb") as f:
        recommender = pickle.load(f)
    with open("grocery_database.pkl", "rb") as f:
        groceries = pickle.load(f)
    return price_model, recommender, groceries


try:
    price_model, recommender, groceries = load_models()
except Exception as e:
    st.error(f"Could not load model files: {e}")
    st.info("Check that the .pkl files are next to app.py and that the scikit-learn "
            "version in requirements.txt matches Colab.")
    st.stop()

vectorizer = recommender["vectorizer"]
meal_vectors = recommender["meal_vectors"]
meals = recommender["meals"].copy()


def predict_prices():
    next_month = (datetime.now().month % 12) + 1
    data = pd.DataFrame({
        "Item_Code": groceries["Item_Code"],
        "Month": [next_month] * len(groceries),
    })
    g = groceries.copy()
    g["Predicted_Price"] = np.round(price_model.predict(data), 2)
    return g


def generate_plan(budget, people, preference, food_preferences, g):
    price_lookup = dict(zip(g["Item"], g["Predicted_Price"]))
    unit_lookup = dict(zip(g["Item"], g["Unit"]))

    query = preference + " " + food_preferences
    sims = cosine_similarity(vectorizer.transform([query]), meal_vectors)[0]
    m = meals.copy()
    m["Similarity"] = sims

    grocery_quantities = {}
    selected = []
    meal_counts = {name: 0 for name in m["Meal"]}

    for day in DAYS:
        for category in CATEGORIES:
            options = m[m["Category"] == category]
            if preference != "Any":
                options = options[options["Preference"] == preference]

            best_meal, best_score, best_qty = None, None, None

            for _, meal in options.iterrows():
                new_qty = grocery_quantities.copy()
                for item, q in meal["Ingredients"].items():
                    new_qty[item] = new_qty.get(item, 0) + q * people

                total_cost = sum(q * price_lookup[i] for i, q in new_qty.items())

                if total_cost <= budget:
                    score = ((1 - meal["Similarity"]) * 100
                             + meal_counts[meal["Meal"]] * 20
                             + (total_cost / budget) * 10)
                    if best_score is None or score < best_score:
                        best_score, best_meal, best_qty = score, meal, new_qty

            if best_meal is not None:
                grocery_quantities = best_qty
                meal_counts[best_meal["Meal"]] += 1
                selected.append([day, category, best_meal["Meal"],
                                 best_meal["Ingredients"], best_meal["Preparation_Time"]])
            else:
                selected.append([day, category, "No meal available", {}, 0])

    grocery_df = pd.DataFrame(
        [[item, round(q, 2), unit_lookup[item], price_lookup[item],
          round(q * price_lookup[item], 2)] for item, q in grocery_quantities.items()],
        columns=["Grocery", "Quantity", "Unit", "Predicted Price (Rs)", "Total Price (Rs)"],
    )

    details = []
    for day, category, name, ingredients, prep in selected:
        if name == "No meal available":
            continue
        cost = sum(q * people * price_lookup[i] for i, q in ingredients.items())
        ing_text = ", ".join(f"{i}: {round(q * people, 2)}" for i, q in ingredients.items())
        details.append([day, category, name, ing_text, round(cost, 2), prep])

    details_df = pd.DataFrame(
        details,
        columns=["Day", "Category", "Meal", "Ingredients", "Estimated Cost (Rs)", "Prep Time (min)"],
    )

    meal_df = pd.DataFrame(selected, columns=["Day", "Category", "Meal", "Ingredients", "Prep"])
    schedule = meal_df.pivot(index="Day", columns="Category", values="Meal")
    schedule = schedule.reindex(index=DAYS, columns=CATEGORIES)

    unfilled = int((meal_df["Meal"] == "No meal available").sum())
    return schedule, grocery_df, details_df, unfilled


# ---------------- UI ----------------
st.title("🍽️ SmartBite")
st.caption("Smart Weekly Grocery Budget & Meal Planner (Machine Learning)")

with st.sidebar:
    st.header("Your Details")
    budget = st.number_input("Weekly budget (Rs)", min_value=0.0, value=5000.0, step=100.0)
    people = st.number_input("Number of people", min_value=1, value=2, step=1)
    preference = st.selectbox("Food preference", ["Any", "Vegetarian", "Non-Vegetarian"])
    food_preferences = st.text_input("Preferred foods", placeholder="e.g. rice, eggs, lentils")
    generate = st.button("Generate Plan", type="primary", use_container_width=True)

g = predict_prices()

if generate:
    if budget <= 0:
        st.error("Budget must be greater than zero.")
    else:
        schedule, grocery_df, details_df, unfilled = generate_plan(
            budget, int(people), preference, food_preferences.strip(), g
        )
        total = grocery_df["Total Price (Rs)"].sum() if not grocery_df.empty else 0.0

        c1, c2, c3 = st.columns(3)
        c1.metric("Weekly Budget", f"Rs. {budget:,.0f}")
        c2.metric("Total Grocery Cost", f"Rs. {total:,.2f}")
        c3.metric("Remaining", f"Rs. {budget - total:,.2f}")

        if unfilled > 0:
            st.warning(f"{unfilled} meal slot(s) could not be filled. Try increasing the budget.")
        else:
            st.success("All meal slots successfully planned!")

        tab1, tab2, tab3, tab4 = st.tabs(
            ["📅 Weekly Schedule", "🛒 Shopping List", "🍲 Meal Details", "📈 Predicted Prices"]
        )
        with tab1:
            st.dataframe(schedule, use_container_width=True)
        with tab2:
            if grocery_df.empty:
                st.info("No groceries selected.")
            else:
                st.dataframe(grocery_df, use_container_width=True, hide_index=True)
                st.download_button("Download shopping list (CSV)",
                                   grocery_df.to_csv(index=False),
                                   "shopping_list.csv", "text/csv")
        with tab3:
            if details_df.empty:
                st.info("No meals available.")
            else:
                st.dataframe(details_df, use_container_width=True, hide_index=True)
        with tab4:
            st.dataframe(
                g[["Item", "Price", "Predicted_Price"]].rename(
                    columns={"Price": "Base Price", "Predicted_Price": "Predicted Next Month"}),
                use_container_width=True, hide_index=True)
else:
    st.info("Enter your details in the sidebar and click **Generate Plan**.")
    st.subheader("Predicted grocery prices for next month")
    st.dataframe(g[["Item", "Price", "Predicted_Price"]],
                 use_container_width=True, hide_index=True)