# molecular-active-learning
Active learning, generator, and ML lives here.


# Generator
The generator should be called at the start of every MD pipline to provide a ligand. 
```bash
python run_generator.py -i -s /tmp/ -n 1 -o <unique place for pipeline to find output>
```

# Aggregator
```bash
python learner_agg.py -i ~/Model-generation/output/ -o ./ -s ~/Model-generation/input/john_smiles_kinasei.smi 
```

After some learning stage

# Learner

For testing one can run:
```bash
python run_learner.py -f -o tmp/ --data_path tmp/fake_data/ --smiles_file sample_input.csv
```