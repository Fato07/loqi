# Website Deployment Guide

## GitHub Pages Deployment

The website is configured to deploy automatically via GitHub Actions when changes are pushed to the `main` branch.

### Initial Setup (One-time)

1. **Enable GitHub Pages in Repository Settings:**
   - Go to your repository: https://github.com/Fato07/loqi
   - Navigate to **Settings** → **Pages**
   - Under **Source**, select **GitHub Actions**
   - Save the settings

2. **Push the workflow file:**
   ```bash
   git push origin main
   ```

### Automatic Deployment

Once enabled, the website will automatically deploy when:
- Changes are pushed to the `main` branch in the `website/` folder
- The deployment workflow file is updated

### Manual Deployment

You can also trigger a manual deployment:
- Go to **Actions** tab in your repository
- Select **Deploy Website to GitHub Pages** workflow
- Click **Run workflow**

### Website URL

After the first successful deployment, your website will be available at:
```
https://fato07.github.io/loqi/
```

### Troubleshooting

- Check the **Actions** tab to see deployment status
- Ensure GitHub Pages is enabled in repository settings
- Verify the workflow file exists at `.github/workflows/deploy.yml`
