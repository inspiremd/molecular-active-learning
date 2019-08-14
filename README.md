# molecular-active-learning
Active learning, generator, and ML lives here.


# Generator
The generator is run initally with -i flag to indicate no uncertainty sampling. The generator uses the -o directory to store 
files to maintain state across invocations.
```bash
python generator/run_generator.py -i -n tasks -o ../generator_store/ -d ../data.csv
```

After some learning stage

# Learner
The learner is called to perform a full train, an online update, or to continue training via -f, -u, or -c flags. The learner uses the -o
directory to maintain state across invocations.
```bash
python learning/main.py -f --data_path  /../path -o ../learner_store/
```