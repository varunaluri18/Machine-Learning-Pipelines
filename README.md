A machine learning pipeline is a way to codify and automate the workflow it takes to produce a machine learning model. 
Machine learning pipelines consist of multiple sequential steps that do everything from data extraction and preprocessing to model training and deployment.

The Benefits of a Machine Learning Pipeline:

It is beneficial to look at the stages which many data science teams go through to understand the benefits of a machine learning pipeline. Implementing the first machine learning models tends to be very problem-oriented, and data scientists focus on producing a model to solve a single business problem, for example, classifying images.

What to Consider when Building a Machine Learning Pipeline?
As stated above, the purpose is to increase the iteration cycle and confidence. Your starting point may vary; for example, you might have already structured your code. The following four steps are an excellent way to approach building an ML pipeline:

Build every step into reusable components.
Consider all the steps that go into producing your machine learning model. Start with how the data is collected and preprocessed, and work your way from there. It’s generally encouraged to limit each component’s scope to make it easier to understand and iterate.
Don’t forget to codify tests into components.
Testing should be considered an inherent part of the pipeline. If you, in a manual process, do some sanity checks on how the input data and the model predictions should look like, you should codify this into a pipeline. A pipeline gives opportunities to be much, much more thorough with testing as you will not have to perform them manually each time.
Tie your steps together.
There are many ways to handle the orchestration of a machine learning pipeline, but the principles remain the same. You define the order in which the components are executed and how inputs and outputs run through the pipeline. We, of course, recommend using Valohai for building your pipeline. The next section is a short overview of how to build a pipeline with Valohai.
Automate when needed.
While building a pipeline already introduces automation as it handles the running of subsequent steps without human intervention, for many, the ultimate goal is also to automatically run the machine learning pipeline when specific criteria are met. For example, you may monitor model drift in production to trigger a re-training run or – simply do it more periodically, like daily.
Depending on your specific use case, your final machine learning pipeline might look different. For example, you might train, evaluate and deploy multiple models in the same pipeline. There are common components that are similar in most machine learning pipelines.


