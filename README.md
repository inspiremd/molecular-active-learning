# molecular-active-learning
Active learning, generator, and ML lives here.


# Generator
The generator is run initally with -i flag to indicate no uncertainty sampling. The generator uses the -o directory to store 
files to maintain state across invocations.
```bash
python run_generator.py -i -n tasks -o tmp/generator_store/ -d ../data.csv
```

After some learning stage

# Learner

For testing one can run:
```bash
 python run_learner.py -f -o tmp/learner_store/ --data_path tmp/fake_data/
```