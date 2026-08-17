# AxiPulseAI System (v6.0)

> **Specification vs implementation.** The "Rules for simulator" section below
> is a *V2.3 specification* of intended operational behavior. The production
> engine (**ForecastAI**) implements OH/NPS using **trained ML models**, not a
> rule-based simulator. Those rules are not an executable formula layer. See
> [`docs/FORECASTAI_ARCHITECTURE.md`](docs/FORECASTAI_ARCHITECTURE.md).

## Column Order (Critical)
**Targets first, then Actuals.**
- Targets: `target_quality`, `target_competency`, `target_attendance`, `target_release_rate`, `target_transfer_rate`
- Actuals: `actual_quality`, `actual_competency`, `actual_attendance`, `actual_release_rate`, `actual_transfer_rate`
- Other: `total_calls_received`, `operational_intelligence_factor`, `issue_type_*`, `operational_health`

## Setup
1. Install Python 3.13
2. `pip install -r requirements.txt`

## Usage
- Train: `python predict_cli.py train data.csv`
- Predict: `python predict_cli.py predict`
- Reverse: `python predict_cli.py reverse --target 105 --factors actual_quality`
- API: `uvicorn api.main:app --reload`

## Codes
python predict_cli.py train data/training_data.csv --save models/test_model.pkl --train the model
python predict_cli.py predict --model models/test_model.pkl --predict for a single day
python predict_cli.py reverse --target 105 --factors actual_quality --model models/test_model.pkl --reverse optimiser
uvicorn api.main:app --reload --for tableau

## AxiPulseAI
source .venv/bin/activate
python predict_cli.py predict --model model.pkl
python predict_cli.py predict --predict model.pkl
python predict_cli.py reverse --target 105 --factors optimising factors --model model.pkl

# AxiPulseAI
python predict_cli.py train-nps_predictor data/your_data.csv --save engine2_model.pkl --train
python predict_cli.py predict-nps --ops-model model.pkl --nps-model engine2_model.pkl --predict

# Predict (Both engines)
python predict_cli.py predict --model model.pkl	///Predict AxiPulseAI (AxiPulseAI) – interactive daily/monthly
python predict_cli.py predict --model model.pkl ///Show only the selected model (no leaderboard)
python predict_cli.py predict-nps --ops-model model.pkl --nps-model engine2_model.pkl	///Predict tomorrow’s NPS (uses both engines) – interactive
python predict_cli.py leaderboard-nps --ops-model model.pkl --nps-model engine2_model.pkl	///Show NPS leaderboard for all models – interactive

# Rules for simluator
Rule 1 - The target release rate will never go below 50%,
Rule 2 - The target quality will never go below 60
Rule 3 - The target transfer rate will never go beyond 20%
Rule 4 - The operation can go beyond and belw +100, 15% of the times it goes both ways
Rule 5 - The total surveys will always be < 15% of total calls 
Rule 6 - Promoter count will minimum be 8 times dertacters + pssives
Rule 7 - Detracters can never be > 8% of released calls
Rule 8 - Passives can never be more than 10% of total released calls
Rule 9 - Mondays and Tuesday i.e 1 then diminishes 0.85,0.8,0.7.5 will have the most calls and then diminshes
RUle 10 - Same as rule 9 but weekly at the start of week
Rule 11 - Saturday and Sunday are holiday
Rule 12 - All 3 factors business_intelligence_factor,operational_intelligence_factor and member_intelligence_factor are default -2%
Rule 13 - 

Category	% of released calls
   survey Rate:     (Range: <10%) of total released calls 
   Promoters:       (Range: <99%  of surveys 
   Passives:        (Range: <2.5%) of surveys 
   Detractors:      (Range: <1.5%) of surveys 
   
complexity increases the detracters and passives by 20% on thse calls only 20% high complex calls gets transferred
competency in 95%=60% release rate
agent should be working in backend no mention of them anywhere in front end
operational health affect all other factor independently
nps b/w 81.5-82.5 is normal when every kpi is met
competency target is 93%
nps vary widely only 10% in first week 1 and 10% in last week
complex tasks have above normal chances of detractors
more handle time leads to promoters 50% or times and detracter/passives 50% of time

call volume decrease by 50% in last week
call volume increases by 20%
noramla call volume is 2000

transfer rate directly impeads the release rate more transfer less releasd calls
queue pressure doesnt matter

