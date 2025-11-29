Week 9 Assignment **Rubric** **Details**<img src="./gwt1wqch.png"
style="width:0.38297in;height:0.39253in" />

> **i(a)**
>
> 5 possible points (11.11%)
>
> **.**
>
> Correctly load and use the new dataset. Brieﬂy describe the dataset
> (e.g., what does it appear to contain? What is the vocabulary size?
> What is the **0** **–** **5** length?). Do the same for the other two
> datasets in the folder.
>
> **i(b)**
>
> 5 possible points (11.11%)
>
> **.**
>
> Successfully downsize below 1 million parameters. Propose and motivate
> a reasonable downsizing strategy. It's crucial that some reasoning is
> carried **0** **–** **5** out to decide which parameters to downsize.
>
> **i(c)**
>
> 10 possible points (22.22%)
>
> **.**
>
> Link the previous observations in i(b) with the numerical results. Are
> the results as expected? What choice appears to be most eﬀective?
> Mention whether the model is overﬁtting (e.g., by considering both
> train and validation loss), and carry out a qualitative assessment of
> the generated output.
>
> Two diﬀerent ways of downsizing must be provided.

**0** **–** **10**

**i(d)**

5 possible points (11.11%)

> **.**
>
> Explore and describe in the report how the inclusion of the bias terms
> in the self-attention layers impacts the transformer model. Penalise
> responses
>
> that are only general, and reward general statements that are then
> connected with the speciﬁc implementation in this exercise. Note that
> the response **0** **–** **5** should include both a general
> description of what the bias terms do, why using them or not in
> general, and considerations on this speciﬁc case/
>
> architecture

**i(e)**

5 possible points (11.11%)

> **.**
>
> Same as i(d) but for skip connections.

**0** **–** **5**

**ii(a)**

10 possible points (22.22%)

> **.**
>
> Select the best model from part (i). Calculate the test loss on
> input_childSpeech_testSet.txt. (this is an unseen portion of the data,
> not used in part (i) ) **0** **–** **10** Report it and comment on it.
> Is it good, bad? Why? It will be crucial to use a baseline of some
> kind (e.g., dummy model).

**ii(b)**

5 possible points (11.11%)

> **.**
>
> Same as for ii(a), but on a dataset with a register that is
> inconsistent with childSpeech (a model ﬁt on child speech is being
> tested on text from **0** **–** **5** Shakespeare). The evaluation
> should be much worse than for ii(a). Explanations should be provided.
