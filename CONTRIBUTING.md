# Contributing to Funhouse

## Issues
Please file a [GitHub issue](https://github.com/run-house/funhouse/issues) if you encounter a bug,
or would like to request or add a new example.

## Updates to Existing Examples

If you would like to submit a bug-fix or improve an existing example, please submit a pull request
following the [development process](#development-process) below.

## New Examples

For inspiration on what example(s) to add, you can take a look at our
[GitHub issues](https://github.com/run-house/funhouse/issues), or join our funhouse channel on
[Discord](https://discord.gg/RnhB6589Hs), where we will post sample applications that you can add!
If you choose to add one of those, please comment on the issue that you are adding it, so others are
aware that you are working on it.

If there is a specific example that you would like to add that is not listed in our GitHub issues, please
first create a new issue describing the example you are adding, then go ahead and follow the
[development process](#development-process) below to add the example.

For all new examples, if applicable, please provide a screenshot or example of the completed output
in your pull request description.

## Development Process
1. Fork the Github repository, and then clone the forked repo to local.
```
git clone git@github.com:<your username>/funhouse.git
cd funhouse
git remote add upstream https://github.com/run-house/funhouse.git
```

2. Create a new branch for your development changes:
```
git checkout -b branch-name
```

3. Install Runhouse
```
pip install runhouse
```

4. Develop changes to funhouse.

5. Add, commit, and push your changes. Create a "Pull Request" on GitHub to submit the changes for
review. Please provide a summary and result of your changes in the PR description.
