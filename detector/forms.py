from django import forms


class UploadForm(forms.Form):
    file = forms.FileField(label="CSV file", widget=forms.ClearableFileInput(attrs={"accept": ".csv"}))
    name = forms.CharField(label="Name", max_length=200, required=False,
                           help_text="Defaults to the file name.")

    def clean_file(self):
        f = self.cleaned_data["file"]
        if not f.name.lower().endswith(".csv"):
            raise forms.ValidationError("Upload a .csv file.")
        return f
