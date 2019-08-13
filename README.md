# molecular-active-learning
Active learning, generator, and ML lives here.


# Generator
The generator is run initally with -i flag to indicate no uncertainty sampling
```bash
python generator/run_generator.py -i -n tasks
```

# Learner
The learner is called to perform a full train, an online update, or to continue training via -f, -u, or -c flags.
```bash
python learning/main.py -f --data_path  /../path 
```