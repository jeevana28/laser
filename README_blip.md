Run the following commands 
```bash
1. python -m venv .venv   (python version >= 3.7 and <= 3.11)
2. source .venv/bin/activate
3. cd coco
4. wget http://images.cocodataset.org/zips/val2014.zip
5. unzip val2014.zip
6. cd ..
7. pip install -r requirements.txt
8. python src/blip_exper.py
