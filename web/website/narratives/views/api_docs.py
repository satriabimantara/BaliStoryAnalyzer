from django.shortcuts import render


def index(request):
    template = "narratives/api_docs/README-balistoryanalyzer.html"
    context = {
        'title_page': 'API Documentation'
    }
    return render(request, template, context)
