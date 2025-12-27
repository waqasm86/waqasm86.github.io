# Analytics Setup Guide for waqasm86.github.io

## Google Analytics 4 Setup

### Step 1: Create GA4 Property

1. Go to https://analytics.google.com
2. Click "Admin" (gear icon)
3. Click "+ Create Property"
4. Property name: `waqasm86.github.io`
5. Select timezone and currency
6. Click "Next"
7. Select industry and business size
8. Select "Get baseline reports" objectives
9. Click "Create"

### Step 2: Add Tracking Code

Once you have your Measurement ID (G-XXXXXXXXXX), add to `mkdocs.yml`:

```yaml
extra:
  analytics:
    provider: google
    property: G-XXXXXXXXXX  # Replace with your actual ID
```

### Step 3: Verify Installation

1. Rebuild site: `mkdocs build`
2. Deploy to GitHub Pages
3. Visit https://waqasm86.github.io
4. Check real-time reports in GA4
5. Verify tracking is working

## Google Search Console Setup

### Step 1: Add Property

1. Go to https://search.google.com/search-console
2. Click "+ Add Property"
3. Select "URL prefix"
4. Enter: `https://waqasm86.github.io`
5. Click "Continue"

### Step 2: Verify Ownership

For GitHub Pages, use HTML tag method:

1. Copy the meta tag provided
2. Add to custom HTML override (if needed)
3. Or use DNS verification (recommended)

**DNS Verification:**
- Add TXT record to your DNS (if using custom domain)
- For github.io subdomain, HTML tag method works best

### Step 3: Submit Sitemap

1. Once verified, go to "Sitemaps" in left menu
2. Enter sitemap URL: `https://waqasm86.github.io/sitemap.xml`
3. Click "Submit"
4. Wait for Google to crawl (24-48 hours)

### Step 4: Request Indexing

1. Go to "URL Inspection"
2. Enter each important page:
   - https://waqasm86.github.io/
   - https://waqasm86.github.io/llcuda/
   - https://waqasm86.github.io/about/
   - https://waqasm86.github.io/contact/
3. Click "Request Indexing" for each

## Bing Webmaster Tools Setup

### Step 1: Sign Up

1. Go to https://www.bing.com/webmasters
2. Sign in with Microsoft account
3. Click "+ Add Site"
4. Enter: `https://waqasm86.github.io`

### Step 2: Import from Google

1. Select "Import from Google Search Console"
2. Authorize Bing to access GSC data
3. Select waqasm86.github.io
4. Click "Import"

### Step 3: Verify & Submit Sitemap

1. Verify ownership (usually auto-verified from GSC)
2. Go to "Sitemaps"
3. Submit: `https://waqasm86.github.io/sitemap.xml`

## PyPI Analytics (llcuda package)

### Using pypistats

Install pypistats:
```bash
pip install pypistats
```

Check recent downloads:
```bash
pypistats recent llcuda
```

Check overall downloads:
```bash
pypistats overall llcuda
```

Check by Python version:
```bash
pypistats python_major llcuda
```

Check by system:
```bash
pypistats system llcuda
```

### Using PyPI Stats Website

Visit: https://pypistats.org/packages/llcuda

Shows:
- Daily download trends
- Python version breakdown
- Operating system distribution
- Download growth over time

## GitHub Analytics

### Repository Insights

1. Go to https://github.com/waqasm86/llcuda
2. Click "Insights" tab
3. View:
   - Traffic (views, clones, referring sites)
   - Commits activity
   - Code frequency
   - Contributors
   - Dependency graph

### Star History

Track star growth:
- https://star-history.com/#waqasm86/llcuda
- https://star-history.com/#waqasm86/Ubuntu-Cuda-Llama.cpp-Executable

## Custom Tracking Events

### In mkdocs.yml (GA4)

Track specific user actions:

```yaml
extra:
  analytics:
    provider: google
    property: G-XXXXXXXXXX
    feedback:
      title: Was this page helpful?
      ratings:
        - icon: material/emoticon-happy-outline
          name: This page was helpful
          data: 1
          note: Thanks for your feedback!
        - icon: material/emoticon-sad-outline
          name: This page could be improved
          data: 0
          note: Thanks for your feedback!
```

### Custom Events to Track

In Google Analytics 4:
1. Downloads (llcuda package links)
2. External links (GitHub, PyPI)
3. Video plays (if embedded)
4. Documentation searches
5. PDF downloads (resume)

## Monitoring Dashboard

Create a weekly dashboard with:

### Key Metrics
- **Users**: Total unique visitors
- **Sessions**: Total visits
- **Page views**: Total pages viewed
- **Bounce rate**: Percentage leaving after one page
- **Avg. session duration**: Time on site

### Traffic Sources
- Organic search
- Direct
- Referral (which sites)
- Social
- GitHub

### Top Pages
1. Homepage
2. llcuda documentation
3. About page
4. Contact
5. Specific tutorials

### Search Console Metrics
- Impressions
- Clicks
- CTR
- Average position
- Top queries
- Top pages

## Weekly Report Checklist

Every Monday:
- [ ] Check GA4 for weekly traffic
- [ ] Review GSC performance (clicks, impressions)
- [ ] Check PyPI downloads (pypistats recent llcuda)
- [ ] Review GitHub Insights traffic
- [ ] Monitor keyword rankings
- [ ] Check for new backlinks
- [ ] Review top landing pages
- [ ] Analyze bounce rate by page

## Monthly Deep Dive

First Monday of each month:
- [ ] Compare month-over-month growth
- [ ] Analyze top traffic sources
- [ ] Review search query trends
- [ ] Check conversion rates (GitHub stars, PyPI installs)
- [ ] Analyze user behavior flow
- [ ] Review exit pages
- [ ] Update keyword strategy
- [ ] Plan content based on popular pages

## Goals & Conversions

### Set up in GA4:

1. **GitHub Visit**
   - Event: click
   - URL contains: github.com/waqasm86

2. **PyPI Visit**
   - Event: click
   - URL contains: pypi.org/project/llcuda

3. **Email Contact**
   - Event: click
   - URL contains: mailto:waqasm86

4. **Documentation Views**
   - Event: page_view
   - Page path: /llcuda/*

## Privacy & GDPR

### Cookie Consent (if needed)

If your site has European visitors, consider adding cookie consent.

For MkDocs Material theme:
```yaml
extra:
  consent:
    title: Cookie consent
    description: >-
      We use cookies to recognize your repeated visits and preferences, as well
      as to measure the effectiveness of our documentation and whether users
      find what they're searching for. With your consent, you're helping us to
      make our documentation better.
```

### Privacy Policy

Create `docs/privacy.md`:
- What data you collect
- How you use it
- Third-party services (GA4)
- User rights
- Contact information

Add to nav in mkdocs.yml:
```yaml
nav:
  - ...
  - Privacy: privacy.md
```

## Troubleshooting

### GA4 not tracking
1. Check measurement ID is correct
2. Verify site is rebuilt and deployed
3. Check browser console for errors
4. Use GA4 DebugView to see real-time events
5. Disable ad blockers for testing

### Search Console not indexing
1. Check robots.txt allows crawling
2. Submit sitemap again
3. Request indexing via URL Inspection
4. Wait 48-72 hours
5. Check for manual actions

### No traffic showing
1. Wait 24-48 hours for data
2. Check if GA4 is in test mode
3. Verify correct property selected
4. Check date range in reports
5. Clear browser cache

## Recommended Tools

### Free
- Google Analytics 4
- Google Search Console
- Bing Webmaster Tools
- GitHub Insights
- pypistats
- Ubersuggest (limited)

### Paid (Optional)
- Ahrefs: $99/mo
- SEMrush: $119/mo
- Hotjar: $39/mo (heatmaps)
- Plausible: $9/mo (privacy-friendly analytics)

## Next Steps

1. ✅ Set up Google Analytics 4
2. ✅ Verify Google Search Console
3. ✅ Add Bing Webmaster Tools
4. ✅ Configure custom events
5. ✅ Set up weekly reporting
6. ✅ Create monitoring dashboard
7. ✅ Track conversions
8. ✅ Add privacy policy (if needed)

---

**Remember**: Analytics are tools for insight, not vanity metrics. Focus on:
- How users find your content
- What content resonates
- Where users drop off
- How to improve user experience

**Goal**: Use data to make informed decisions about content creation and SEO strategy.
