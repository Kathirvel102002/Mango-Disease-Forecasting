import pandas as pd
import numpy as np
import joblib
import os
import ipywidgets as widgets
from IPython.display import display, clear_output

# ==========================================================
# 1. CREATE WIDGETS (USER INTERFACE)
# ==========================================================

disease_dropdown = widgets.Dropdown(
    options=['Select Disease...', 'Leaf Anthracnose', 'Black Banded', 'Red Rust', 'Die Back', 'Sooty Mould'],
    value='Select Disease...',
    description='Disease:',
    style={'description_width': 'initial'}
)

rf = widgets.FloatText(value=25.0, description='Rainfall (RF) mm:', style={'description_width': 'initial'})
rd = widgets.IntText(value=2, description='Rainy Days (RD):', style={'description_width': 'initial'})
rh = widgets.FloatText(value=80.0, description='Humidity (RH) %:', style={'description_width': 'initial'})
tmax = widgets.FloatText(value=34.0, description='Max Temp (°C):', style={'description_width': 'initial'})
tmin = widgets.FloatText(value=25.0, description='Min Temp (°C):', style={'description_width': 'initial'})
current_disease = widgets.FloatText(value=15.0, description='Current Disease (%):', style={'description_width': 'initial'})
week = widgets.IntText(value=24, description='Week Number (1-52):', style={'description_width': 'initial'})

predict_btn = widgets.Button(
    description='Forecast Disease Risk',
    button_style='success',
    icon='check'
)

output = widgets.Output()

# Organize the layout into columns
weather_box = widgets.VBox([widgets.HTML("<b>Weather Parameters</b>"), rf, rd, rh, tmax, tmin])
field_box = widgets.VBox([widgets.HTML("<b>Field Parameters</b>"), current_disease, week])
input_area = widgets.HBox([weather_box, field_box])

ui = widgets.VBox([
    widgets.HTML("<h2>🥭 Mango Disease Forecast System</h2>"),
    disease_dropdown,
    input_area,
    predict_btn,
    output
])

# ==========================================================
# 2. PREDICTION LOGIC
# ==========================================================

def predict_clicked(b):
    with output:
        clear_output()
        disease = disease_dropdown.value

        if disease == 'Select Disease...':
            print("⚠️ Please select a disease from the dropdown menu first.")
            return

        print(f"--- ANALYZING {disease.upper()} ---")

        # 1. Background Calculations (Matching your XGBoost requirements)
        T_avg = (tmax.value + tmin.value) / 2
        T_Range = tmax.value - tmin.value
        week_sin = np.sin(2 * np.pi * week.value / 52)
        week_cos = np.cos(2 * np.pi * week.value / 52)

        # 2. Format Data exactly as the model expects
        input_data = pd.DataFrame([[
            rf.value, rd.value, rh.value, tmax.value, tmin.value, 
            T_avg, T_Range, current_disease.value, week_sin, week_cos
        ]], columns=[
            "RF", "RD", "RH", "T_MAX", "T_MIN", "T_avg", "T_Range", "DISEASE", "week_sin", "week_cos"
        ])

        # 3. Route to the correct file
        model_files = {
            'Leaf Anthracnose': 'LEAF ANTHRACNOSE_farmer_forecast.pkl',
            'Black Banded': 'BLACK_BANDED_farmer_forecast.pkl',
            'Red Rust': 'RED RUST_farmer_forecast.pkl',
            'Die Back': 'DIE BACK_farmer_forecast.pkl',
            'Sooty Mould': 'SOOTY MOULD_farmer_forecast.pkl'
        }
        
        filename = model_files[disease]

        # 4. Check if file exists, then load and predict
        if not os.path.exists(filename):
            print(f"❌ ERROR: Could not find '{filename}'.")
            print("Did you upload the .pkl files to Colab's file menu on the left?")
            return

        try:
            model = joblib.load(filename)
            forecast = model.predict(input_data)[0]

            # Risk Assessment
            if forecast < 20:
                risk, advice = "Low 🟢", "Routine monitoring is sufficient."
            elif forecast < 40:
                risk, advice = "Moderate 🟡", "Inspect orchard regularly."
            elif forecast < 60:
                risk, advice = "High 🟠", "Start disease management measures."
            else:
                risk, advice = "Epidemic 🔴", "Immediate control measures required."

            # Display Results
            print("==================================================")
            print("               NEXT WEEK FORECAST                 ")
            print("==================================================")
            print(f"Predicted Disease Severity : {forecast:.2f}%")
            print(f"Risk Level                 : {risk}")
            print(f"Recommendation             : {advice}")
            print("==================================================")

        except Exception as e:
            print(f"❌ ERROR: {str(e)}")

# Attach the click event to the button
predict_btn.on_click(predict_clicked)

# ==========================================================
# 3. DISPLAY THE APP
# ==========================================================
display(ui)