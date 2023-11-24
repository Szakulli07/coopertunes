# coopertunes
Hub for music machine learning  generating audio

Amazing README coming soon!

# Installation

It is recommended to use conda environment for setup `coopertunes` module. To install conda on your machine follow [this](https://conda.io/projects/conda/en/stable/user-guide/install/linux.html) instruction. If you already have conda installed create a virtual environment"

`conda create -n coopertunes python=3.10`

and activate it:

`conda activate coopertunes`

Clone coopertunes repository:

`git clone git@github.com:Szakulli07/coopertunes.git`
`cd coopertunes`

Before install `coopertunes` module you need to install `pytorch` framework. It is recommended to install version `2.0.1`:

`pip install torch==2.0.1 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118`

Now you need to install `coopertunes` module:

`pip install -e .`

# How should I write my commits?

Release Please assumes you are using [Conventional Commit messages](https://www.conventionalcommits.org/).

The most important prefixes you should have in mind are:

* `fix:` which represents bug fixes, and correlates to a [SemVer](https://semver.org/)
  patch.
* `feat:` which represents a new feature, and correlates to a SemVer minor.
* `feat!:`,  or `fix!:`, `refactor!:`, etc., which represent a breaking change
  (indicated by the `!`) and will result in a SemVer major.
