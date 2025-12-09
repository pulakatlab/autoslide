# DVC Setup

This project uses [DVC (Data Version Control)](https://dvc.org/) to track large files and model artifacts.

## Installation

Install DVC with your preferred storage backend:

```bash
# Base installation
pip install dvc

# For Google Drive storage
pip install dvc[gdrive]

# For S3 storage
pip install dvc[s3]

# For Azure storage
pip install dvc[azure]
```

## Initial Setup

DVC is already initialized in this repository. To configure remote storage:

### Google Drive

```bash
dvc remote add -d myremote gdrive://folder_id
```

### Amazon S3

```bash
dvc remote add -d myremote s3://bucket/path
```

### Azure Blob Storage

```bash
dvc remote add -d myremote azure://container/path
```

## Working with DVC

### Adding Files to DVC

Track large files or directories:

```bash
# Add a large dataset
dvc add data/large_dataset.svs

# Add model artifacts
dvc add artifacts/mask_rcnn_model.pth

# Add entire directories
dvc add data/slides/
```

This creates `.dvc` files that you commit to git instead of the large files.

### Committing Changes

```bash
# Stage the .dvc file
git add data/large_dataset.svs.dvc

# Commit to git
git commit -m "Add large dataset"

# Push data to remote storage
dvc push
```

### Pulling Data

Retrieve data tracked by DVC:

```bash
# Pull all tracked data
dvc pull

# Pull specific file
dvc pull data/large_dataset.svs.dvc
```

## Versioning Models

When you train a new model version:

```bash
# Add the new model
dvc add artifacts/mask_rcnn_model.pth

# Commit the .dvc file
git add artifacts/mask_rcnn_model.pth.dvc
git commit -m "Update model with improved accuracy"

# Push to remote storage
dvc push
```

## Switching Between Versions

```bash
# Checkout a specific commit
git checkout <commit-hash>

# Get the corresponding data
dvc pull
```

## Pipeline Integration

DVC can track pipeline outputs:

```bash
# Track pipeline outputs
dvc add output/predictions/
dvc add output/fibrosis/

# Commit and push
git add output/predictions.dvc output/fibrosis.dvc
git commit -m "Add pipeline outputs"
dvc push
```

## Best Practices

1. **Always use DVC for large files** - Don't commit large files directly to git
2. **Commit .dvc files to git** - These small metadata files track your data
3. **Push after adding** - Run `dvc push` to make data available to collaborators
4. **Pull after checkout** - Run `dvc pull` after switching branches/commits
5. **Use .gitignore** - Ensure tracked files are in .gitignore

## Troubleshooting

### Authentication Issues

For Google Drive:

```bash
dvc remote modify myremote gdrive_use_service_account true
dvc remote modify myremote gdrive_service_account_json_file_path path/to/credentials.json
```

For S3:

```bash
dvc remote modify myremote access_key_id <key>
dvc remote modify myremote secret_access_key <secret>
```

### Cache Issues

Clear DVC cache if needed:

```bash
dvc cache dir  # Show cache location
rm -rf .dvc/cache  # Clear cache (use with caution)
```

## Additional Resources

- [DVC Documentation](https://dvc.org/doc)
- [DVC with Google Drive](https://dvc.org/doc/user-guide/data-management/remote-storage/google-drive)
- [DVC with S3](https://dvc.org/doc/user-guide/data-management/remote-storage/amazon-s3)
