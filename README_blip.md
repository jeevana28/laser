Run the following commands 
```bash
1. python -m venv .venv   (python version >= 3.7 and <= 3.11)
2. source .venv/bin/activate
3. cd coco
4. wget http://images.cocodataset.org/zips/val2014.zip
5. unzip val2014.zip
6. cd ..
7. pip install -r requirements.txt
8. python src/intervention_blip_coco.py --lname "dont" --rate 9.9 --lnum -1
```
(If 8 works fine and .p and .pkl files are created in coco/results/blip_results/BLIP/rank-reduction/dont, proceed with step numver 9, otherwise  open  .venv/lib/python3.10/site-packages/pycocoevalcap/eval.py and comment out line num 45, (remove comma in line 44) and process with step 9)

```bash
9. python src/blip_exper.py
