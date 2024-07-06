## Launch the app locally

From the root directory of the project run:

```bash
python3 app/main.py
```

## Commands to deploy for the first time in Heroku

Initialize a GitHub repo for the project
```bash
git init
git add .
git commit -m "Initial commit"
```

Create a new Heroku app

```bash
heroku create [APP-NAME]
```

## Commands to update Heroku app

1. Commit changes to Git repository
2. Push to Heroku with:

```bash
git push heroku master
```

## Check logs
heroku logs --app [APP-NAME] --tail