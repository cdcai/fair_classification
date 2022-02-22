# Fair classification through linear programming
## Introduction
This repository implements several postprocessing algorithms designed to debias pretrained classifiers. It uses linear programming for everything, including its work with real-valued predictors, and so the bulk of the solving code is an implementation (and extension) of the method presented by Hardt, Price, and Srebro in their [2016 paper](https://arxiv.org/pdf/1610.02413.pdf) on fairness in supervised learning. 

## Methods
### Background
The main goal of any postprocessing method is to take an existing classifier and make it fair for all levels of a protected category, like race or religion. There are a number of ways to do this, but in Hardt, Price, and Srebro's paper, they take an oblivious approach, such that the adjusted classifier (or derived predictor, Y<sub>tilde</sub>) only relies on the joint distribution of the true label (Y), the predicted label (Y<sub>hat</sub>), and the protected attribute (A).

#### Discrete predictor for a binary outcome
When both Y and Y hat are binary, the optimization problem is a linear program based on the conditional probabilities (Y<sub>tilde</sub> = Y<sub>hat</sub>). In this case, two probabilities, (Y<sub>tilde</sub> = 1 | Y<sub>hat</sub> = 1) and (Y<sub>tilde</sub> = 1 | Y<sub>hat</sub> = 0) fully define the distribution of the derived predictor, and so the linear program can be written in a very parismonious 2 * N<sub>groups</sub> variables. The solution will find the topmost-left point of the intersection of the group-specific convex hulls defined by the conditional probabilities in ROC space (illustration below).

<img src="https://github.com/scotthlee/fairness/blob/master/img/nolines.png" width="400" height="300"><img src="https://github.com/scotthlee/fairness/blob/master/img/lines.png" width="400" height="300">

#### Continuous predictor for a binary outcome
When Y is binary but Y<sub>hat</sub> is continuous, Y<sub>hat</sub> must be thresholded before we can examine it for fairness. Hardt, Price, and Srebrbo proposed adding randomness to the selection of threshold for each group to make the resulting predictions fair. Here, we take the arguably more straightforward approach of thresholding the scores first (choosing thresholds that maximize groupwise performance) and then using the linear program as in the discrete case above to solve for the derived predictor. Theoretically, this may be sub-optimal, but practically, it runs fast and works well (illustration below). 

<img src="https://github.com/scotthlee/fairness/blob/master/img/roc_nolines.png" width="400" height="300"><img src="https://github.com/scotthlee/fairness/blob/master/img/roc_lines.png" width="400" height="300">

#### Multiclass outcomes
In our [paper](http://ceur-ws.org/Vol-3087/paper_36.pdf) from the [SafeAI workshop](https://safeai.webs.upv.es) at AAAI 2022, we extend the binary approach above to multiclass outcomes. As before, the solution still uses a linear program to derive the adjusted predictor, but the program's constraints and loss functions change a bit to account for the somewhat expanded definitions of fairness entailed by the added outcome levels. Our implementation supports the same functions as above, including plotting:

<img src="https://github.com/scotthlee/fairness/blob/master/img/strict goal with macro loss.png" width="1000" height="300">

Notice that in this particular example the optima are not truly optimal--they're a bit inside of the bounding convex hull for one outcome, and a bit outside it for the others. A good solution can be hard to find when the number of outcomes is high, when there's lots of class imblanace, or when the disparities between groups are particularly strong. We provide a fairly comprehensive look at these different scenarios in our paper.

### Implementation
Our implementation use two classes, the `BinaryBalancer` and the `MulticlassBalancer`, to perform their respective adjustments. Initializing a balancer with the true label, the predicted label, and the protected attribute will produce a report with the groupwise true- and false-positive rates. The rest of its functionality comes from the `.adjust()` and `.plot()` methods--see the two classes's [docstrings](balancers/__init__.py) for more info!

## Data
For demo purposes, the repository comes with a synthetic dataset, [farm_animals.csv](data/farm_animals.csv), which we created with [data_gen.py](data/data_gen.py). Here are the data elements:

1. `animal`: The kind of farm animal. Options are `cat`, `dog`, and `sheep`. This is the protected attribute A.
2. `action`: The kind of care the animal needs. Options are `feed`, `pet`, and `shear`. This is the true label Y.
3. `pred_action`: The kind of care the farmer thinks the animal needs. This is the predicted label Y<sub>hat</sub>.
4. `shear`: Whether `pred_action` is `shear`.
5. `shear_prob`: The predicted probability that the animal needs a shear. This was generated using different conditional probabilities than the variable `pred_action`, so it will not equal `shear` when thresholded. 

The distirbution of animals is not entirely realistic--a working sheep farmer, for example, would have a much higher ratio of sheep to herding dogs--but the lower class imbalance makes the demo a bit easier to follow.

## Demo
To see the postprocessing algorithms in action, please check out the [demo notebook](demo.ipynb). The notbeook shows off the main features of the two `Balancer` classes and is the best place to start if you've never worked with these kinds of adjustments before. Please note: The modules in `requirements.txt` must be installed before running.

## Citation
If you use this code for a project, please give us a shout out.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4890946.svg)](https://doi.org/10.5281/zenodo.4890946)

## Public Domain Standard Notice
This repository constitutes a work of the United States Government and is not
subject to domestic copyright protection under 17 USC § 105. This repository is in
the public domain within the United States, and copyright and related rights in
the work worldwide are waived through the [CC0 1.0 Universal public domain dedication](https://creativecommons.org/publicdomain/zero/1.0/).
All contributions to this repository will be released under the CC0 dedication. By
submitting a pull request you are agreeing to comply with this waiver of
copyright interest.

## License Standard Notice
The repository utilizes code licensed under the terms of the Apache Software
License and therefore is licensed under ASL v2 or later.

This source code in this repository is free: you can redistribute it and/or modify it under
the terms of the Apache Software License version 2, or (at your option) any
later version.

This source code in this repository is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the Apache Software License for more details.

You should have received a copy of the Apache Software License along with this
program. If not, see http://www.apache.org/licenses/LICENSE-2.0.html

The source code forked from other open source projects will inherit its license.

## Privacy Standard Notice
This repository contains only non-sensitive, publicly available data and
information. All material and community participation is covered by the
[Disclaimer](https://github.com/CDCgov/template/blob/master/DISCLAIMER.md)
and [Code of Conduct](https://github.com/CDCgov/template/blob/master/code-of-conduct.md).
For more information about CDC's privacy policy, please visit [http://www.cdc.gov/privacy.html](http://www.cdc.gov/privacy.html).

## Contributing Standard Notice
Anyone is encouraged to contribute to the repository by [forking](https://help.github.com/articles/fork-a-repo)
and submitting a pull request. (If you are new to GitHub, you might start with a
[basic tutorial](https://help.github.com/articles/set-up-git).) By contributing
to this project, you grant a world-wide, royalty-free, perpetual, irrevocable,
non-exclusive, transferable license to all users under the terms of the
[Apache Software License v2](http://www.apache.org/licenses/LICENSE-2.0.html) or
later.

All comments, messages, pull requests, and other submissions received through
CDC including this GitHub page are subject to the [Presidential Records Act](http://www.archives.gov/about/laws/presidential-records.html)
and may be archived. Learn more at [http://www.cdc.gov/other/privacy.html](http://www.cdc.gov/other/privacy.html).

## Records Management Standard Notice
This repository is not a source of government records, but is a copy to increase
collaboration and collaborative potential. All government records will be
published through the [CDC web site](http://www.cdc.gov).

## Additional Standard Notices
Please refer to [CDC's Template Repository](https://github.com/CDCgov/template)
for more information about [contributing to this repository](https://github.com/CDCgov/template/blob/master/CONTRIBUTING.md),
[public domain notices and disclaimers](https://github.com/CDCgov/template/blob/master/DISCLAIMER.md),
and [code of conduct](https://github.com/CDCgov/template/blob/master/code-of-conduct.md).
