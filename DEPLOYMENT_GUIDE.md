# 🚀 GitHub Repository Setup & Deployment Guide

## Step 1: Create GitHub Repository

### 1a. Create a new repository on GitHub.com
1. Go to https://github.com/new
2. **Repository name:** `bank-marketing-classifier` (or your preferred name)
3. **Description:** Bank Marketing Classification Model with 5 ML algorithms and Streamlit deployment
4. **Visibility:** Public (required for Streamlit Community Cloud)
5. **Initialize with:** No (we'll push existing code)
6. Click **Create repository**

### 1b. Copy your repository URL
```
https://github.com/YOUR_USERNAME/bank-marketing-classifier.git
```

---

## Step 2: Initialize Local Git & Push to GitHub

### 2a. Navigate to project directory
```bash
cd ml_assignment
```

### 2b. Initialize git repository
```bash
git init
git config user.name "Your Name"
git config user.email "your.email@example.com"
```

### 2c. Add all files to git
```bash
git add .
```

### 2d. Create initial commit
```bash
git commit -m "Initial commit: Bank Marketing Classification Models with Streamlit App"
```

### 2e. Add remote origin and push
```bash
git remote add origin https://github.com/YOUR_USERNAME/bank-marketing-classifier.git
git branch -M main
git push -u origin main
```

---

## Step 3: Deploy on Streamlit Community Cloud

### 3a. Go to Streamlit Community Cloud
1. Visit https://share.streamlit.io
2. Click "New app" button

### 3b. Connect GitHub Account
1. Click "Connect GitHub"
2. Authorize Streamlit to access your GitHub account
3. Select your repository: `bank-marketing-classifier`
4. Select branch: `main`
5. Set main file path: `app.py`

### 3c. Configure App Settings (Optional)
- App name: Bank Marketing Classifier
- Advanced settings: Default (no changes needed)

### 3d. Deploy
1. Click "Deploy"
2. Streamlit will build and deploy your app (takes 2-3 minutes)
3. Your app will be live at: `https://share.streamlit.io/YOUR_USERNAME/bank-marketing-classifier/main/app.py`

---

## Step 4: Verify Deployment

### 4a. Test the Deployed App
1. Open your Streamlit app URL
2. Test each feature:
   - ✅ Model metrics display correctly
   - ✅ Manual prediction input works
   - ✅ CSV upload functionality works
   - ✅ Model comparison shows all 5 models
   - ✅ Model dropdown switches between models

### 4b. Verify Data Upload
1. Try uploading `test_data.csv` from the repo
2. Should generate predictions for all 50 rows

---

## Step 5: Final Submission Checklist

### GitHub Repository ✅
- [ ] Repository created and public
- [ ] All files pushed to main branch
- [ ] README.md displays correctly
- [ ] Model files present in model/ directory
- [ ] test_data.csv available
- [ ] requirements.txt complete

### Streamlit Cloud ✅
- [ ] App deployed successfully
- [ ] All 4 interactive features working
- [ ] Model selection dropdown functional
- [ ] CSV upload feature working
- [ ] Metrics display correctly
- [ ] Model comparison charts visible

### Documentation ✅
- [ ] Comprehensive README.md written
- [ ] All 6 metrics explained
- [ ] Model comparison table included
- [ ] Installation instructions provided
- [ ] Dataset description complete

---

## 📋 Submission Links Template

For your assignment submission, provide:

**GitHub Repository Link:**
```
https://github.com/YOUR_USERNAME/bank-marketing-classifier
```

**Streamlit App Link:**
```
https://share.streamlit.io/YOUR_USERNAME/bank-marketing-classifier/main/app.py
```

**Key Files in Repository:**
- `app.py` - Main Streamlit application
- `README.md` - Comprehensive documentation
- `requirements.txt` - All dependencies
- `model/` - Directory with 5 trained models
- `test_data.csv` - Sample test data
- `bank_marketing_full.csv` - Full training dataset

---

## 🐛 Troubleshooting

### Issue: "App deployment failed"
**Solution:**
1. Check requirements.txt for correct versions
2. Ensure all files are committed to git
3. Try redeploying from Streamlit Cloud dashboard

### Issue: "Module not found error"
**Solution:**
1. Verify all packages in requirements.txt are listed
2. Check spelling and capitalization
3. Run `pip install -r requirements.txt` locally to test

### Issue: "Models not loading"
**Solution:**
1. Ensure `model/` directory exists with all .joblib files
2. Verify model names match exactly in app.py
3. Check file paths are relative (not absolute)

### Issue: "CSV upload not working"
**Solution:**
1. Ensure test_data.csv has all required columns
2. Verify categorical columns match training data encoding
3. Test locally first: `streamlit run app.py`

---

## 📞 Support & Questions

If you encounter issues:
1. Check Streamlit documentation: https://docs.streamlit.io/
2. Review GitHub deployment guide: https://docs.github.com/en/github/getting-started-with-github/github-workflow
3. Check scikit-learn docs: https://scikit-learn.org/
4. Contact BITS support for assignment-specific questions

---

## ✅ Final Verification Checklist

Before submitting:

- [ ] GitHub repository is public and accessible
- [ ] All 5 models are trained and saved
- [ ] Streamlit app runs without errors
- [ ] All 4 interactive features are functional
- [ ] README.md includes all required sections
- [ ] Model results show all 6 metrics
- [ ] Model comparison table is comprehensive
- [ ] Dependencies are listed in requirements.txt
- [ ] Test data is available and correct
- [ ] Deployment on Streamlit Community Cloud is successful
- [ ] Both GitHub and Streamlit links are working

---

**Status:** 🟢 Ready for Submission  
**Deadline:** November 8, 2026, 23:59 IST  
**Total Marks:** 15 (10 + 4 + 1)

Good luck with your BITS Pilani assignment! 🎓
