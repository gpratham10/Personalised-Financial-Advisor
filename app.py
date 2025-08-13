import os, uuid
import matplotlib
matplotlib.use("Agg")

from flask import Flask, render_template, request, send_file, jsonify
import joblib
import matplotlib.pyplot as plt
from fpdf import FPDF

# Chatbot integration
from chatbot_engine import generate_chat_response

app = Flask(__name__)
os.makedirs("static", exist_ok=True)

# ------------ Helpers ------------
def safe_div(a, b):
    return (a / b) if b else 0.0

def save_fig(path):
    plt.savefig(path, bbox_inches="tight")
    plt.clf(); plt.close('all')

def explain_choice(inv_type, overspend_final):
    if overspend_final:
        return ("You are overspending. Focus on cutting expenses and building an emergency fund "
                "before investing.")
    if inv_type == "Fixed Deposit":
        return "Very conservative / low savings -> Fixed Deposit offers safety and liquidity."
    if inv_type == "Gold":
        return "Low savings rate and conservative profile -> Gold is a safer wealth preserver."
    if inv_type == "Mutual Fund":
        return "Moderate savings and balanced risk tolerance -> Mutual Fund suits you."
    if inv_type == "Stocks":
        return "High savings and capacity for risk -> Stocks can offer higher long-term returns."
    return "Based on your pattern, start with safe products and review your finances."

def add_img(pdf, path, w=180):
    if not path:
        return
    if pdf.get_y() > 240:
        pdf.add_page()
    pdf.image(path, x=15, w=w)
    pdf.ln(8)

# ------------ Load Models ------------
overspend_model  = joblib.load("lightgbm_overspending_model.pkl")
savings_model    = joblib.load("savings_rate_model_rf_optimized.pkl")
investment_model = joblib.load("lightgbm_behavioral_model.pkl")

# ------------ Routes ------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        form = request.form
        num_keys = [k for k in form if k not in ["Occupation", "City_Tier"]]
        u = {k: float(form[k]) for k in num_keys}
        u["Occupation"] = int(form["Occupation"])
        u["City_Tier"]  = int(form["City_Tier"])

        income = u["Income"]
        if income <= 0:
            return "Income must be > 0."

        # ---- Percentages ----
        pct_rent          = safe_div(u["Rent"],          income)
        pct_insurance     = safe_div(u["Insurance"],     income)
        pct_transport     = safe_div(u["Transport"],     income)
        pct_education     = safe_div(u["Education"],     income)
        pct_groceries     = safe_div(u["Groceries"],     income)
        pct_eating_out    = safe_div(u["Eating_Out"],    income)
        pct_entertainment = safe_div(u["Entertainment"], income)
        pct_miscellaneous = safe_div(u["Miscellaneous"], income)
        pct_utilities     = safe_div(u["Utilities"],     income)
        has_loan          = int(u["Loan_Repayment"] > 0)

        # ---- Savings model (10 feats)
        savings_feats = [
            income, u["Age"], u["Dependents"], u["Occupation"],
            u["Desired_Savings_Percentage"], pct_rent, pct_insurance,
            pct_transport, pct_education, has_loan
        ]
        savings_rate = float(savings_model.predict([savings_feats])[0])
        savings_gap  = u["Desired_Savings_Percentage"] - savings_rate

        # ---- Overspending model (10 feats)
        overspend_feats = [
            income, u["Age"], u["Dependents"], u["Occupation"], u["City_Tier"],
            pct_groceries, pct_eating_out, pct_entertainment, pct_miscellaneous, pct_utilities
        ]
        overspend_ml = int(overspend_model.predict([overspend_feats])[0])

        # Manual overspend
        total_expense = sum([
            u["Rent"], u["Loan_Repayment"], u["Groceries"], u["Transport"], u["Eating_Out"],
            u["Entertainment"], u["Utilities"], u["Insurance"], u["Education"], u["Miscellaneous"]
        ])
        actual_savings     = income - total_expense
        actual_savings_pct = safe_div(actual_savings, income) * 100
        overspend_math     = int(actual_savings < 0)
        overspend_final    = 1 if (overspend_ml or overspend_math) else 0

        # ---- Investment suggestion
        inv_probs = [0, 0, 0, 0]
        inv_map   = {0: "Fixed Deposit", 1: "Gold", 2: "Mutual Fund", 3: "Stocks"}

        if overspend_final:
            investment_type = "None (reduce expenses first)"
        else:
            inv_feats = [
                income, u["Age"], u["Dependents"], u["Occupation"], u["City_Tier"],
                u["Desired_Savings_Percentage"], pct_eating_out, pct_entertainment,
                pct_miscellaneous, pct_insurance, pct_transport,
                actual_savings_pct, savings_gap
            ]
            inv_label = int(investment_model.predict([inv_feats])[0])
            if hasattr(investment_model, "predict_proba"):
                inv_probs = investment_model.predict_proba([inv_feats])[0].tolist()
            else:
                inv_probs[inv_label] = 1.0
            investment_type = inv_map.get(inv_label, "Fixed Deposit")

        explanation = explain_choice(investment_type, overspend_final)

        # ---- Charts ----
        exp_map = {
            "Rent": u["Rent"], "Insurance": u["Insurance"], "Transport": u["Transport"],
            "Education": u["Education"], "Loan_Repayment": u["Loan_Repayment"],
            "Groceries": u["Groceries"], "Eating_Out": u["Eating_Out"],
            "Entertainment": u["Entertainment"], "Utilities": u["Utilities"],
            "Miscellaneous": u["Miscellaneous"]
        }

        cats = list(exp_map.keys())
        vals = [float(v) for v in exp_map.values()]
        pct_vals_income = [safe_div(v, income) * 100 for v in vals]

        uid = uuid.uuid4().hex
        pie_path   = f"static/{uid}_pie.png"
        top_path   = f"static/{uid}_top.png"
        pct_path   = f"static/{uid}_pct.png"
        gap_path   = f"static/{uid}_gap.png"
        nw_path    = f"static/{uid}_nw.png"
        stack_path = f"static/{uid}_stack.png"
        prob_path  = f"static/{uid}_prob.png"

        # Create charts
        plt.figure(figsize=(6, 6))
        plt.pie(vals, labels=cats, autopct=lambda p: f'{p:.0f}%', startangle=90)
        plt.title("Spending Distribution")
        save_fig(pie_path)

        top_pairs = sorted(exp_map.items(), key=lambda x: x[1], reverse=True)[:5]
        top_labels, top_amounts = zip(*top_pairs) if top_pairs else ([], [])
        plt.figure(figsize=(7, 4))
        plt.barh(top_labels, top_amounts, color='#ff8c69')
        plt.gca().invert_yaxis()
        plt.xlabel("Amount (INR)")
        plt.title("Top Spending Categories")
        save_fig(top_path)

        plt.figure(figsize=(8, 4))
        plt.bar(cats, pct_vals_income, color='#2e8b57')
        plt.xticks(rotation=45, ha='right')
        plt.ylabel("% of Income")
        plt.title("Category-wise % of Income")
        save_fig(pct_path)

        plt.figure(figsize=(6, 4))
        actual_savings_amt  = max(actual_savings, 0)
        desired_savings_amt = income * u["Desired_Savings_Percentage"] / 100
        plt.bar(["Actual"], [total_expense], color='#ff9999', label="Expenses")
        plt.bar(["Actual"], [actual_savings_amt], bottom=[total_expense],
                color='#99ccff', label="Actual Savings")
        plt.bar(["Desired"], [income - desired_savings_amt], color='#ff9999')
        plt.bar(["Desired"], [desired_savings_amt], bottom=[income - desired_savings_amt],
                color='#f7d26a', label="Desired Savings")
        plt.ylabel("Amount (INR)")
        plt.title("Income Split: Expenses + Savings")
        plt.legend()
        save_fig(stack_path)

        needs_amt = sum(exp_map[k] for k in ["Rent", "Loan_Repayment", "Utilities", "Insurance", "Transport"])
        wants_amt = sum(exp_map[k] for k in ["Entertainment", "Eating_Out", "Miscellaneous"])
        plt.figure(figsize=(5, 4))
        plt.bar(["Needs", "Wants"], [needs_amt, wants_amt], color=['green', 'orange'])
        plt.ylabel("Amount (INR)")
        plt.title("Needs vs Wants")
        save_fig(nw_path)

        prob_text = ""
        if not overspend_final:
            plt.figure(figsize=(6, 4))
            labels_probs = ["Fixed Deposit", "Gold", "Mutual Fund", "Stocks"]
            plt.bar(labels_probs, [p * 100 for p in inv_probs], color='#6a5acd')
            plt.ylabel("Probability (%)")
            plt.title("Investment Recommendation Probabilities")
            plt.ylim(0, 100)
            for i, p in enumerate(inv_probs):
                plt.text(i, p * 100 + 1, f"{p*100:.1f}%", ha='center', fontsize=9)
            save_fig(prob_path)
            prob_text = ", ".join([f"{lbl}: {p*100:.1f}%" for lbl, p in zip(labels_probs, inv_probs)])
        else:
            prob_path = ""

        # ---- PDF ----
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", "B", 16)
        pdf.cell(200, 10, "Financial Report", ln=True)
        pdf.set_font("Arial", size=12)
        pdf.cell(0, 8, f"Income: INR {int(income)}", ln=True)
        pdf.cell(0, 8, f"Savings Rate: {savings_rate:.2f}%", ln=True)
        pdf.cell(0, 8, f"Savings Gap: {savings_gap:.2f}%", ln=True)
        pdf.cell(0, 8, f"Overspending: {'Yes' if overspend_final else 'No'}", ln=True)
        pdf.cell(0, 8, f"Recommended Investment: {investment_type}", ln=True)
        if prob_text:
            pdf.multi_cell(0, 8, f"Model probabilities -> {prob_text}")
        pdf.multi_cell(0, 8, f"Reason: {explanation}")

        # Charts in PDF
        for path in [pie_path, top_path, pct_path, stack_path, nw_path, prob_path]:
            add_img(pdf, path)
        pdf.output("static/result.pdf")

        return render_template(
            "result.html",
            savings_rate=f"{savings_rate:.2f}",
            savings_gap=f"{savings_gap:.2f}",
            overspending_status="Yes" if overspend_final else "No",
            investment_type=investment_type,
            explanation=explanation,
            prob_text=prob_text,
            pie_chart=pie_path,
            top_chart=top_path,
            pct_chart=pct_path,
            gap_chart=gap_path,
            nw_chart=nw_path,
            stack_chart=stack_path,
            prob_chart=prob_path
        )

    except Exception as e:
        return f"Error during prediction: {str(e)}"

@app.route("/download")
def download_report():
    return send_file("static/result.pdf", as_attachment=True)

# ------------ Chatbot Endpoint ------------
@app.route("/chat", methods=["POST"])
def chat():
    try:
        user_msg = request.json.get("message", "")
        response = generate_chat_response(user_msg)
        return jsonify({"response": response})
    except Exception as e:
        return jsonify({"response": f"Error: {str(e)}"})

if __name__ == "__main__":
    app.run(debug=False)
