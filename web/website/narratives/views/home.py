from django.shortcuts import render


def index(request):
    template = "narratives/index.html"
    context = {
        'title_page': ''
    }
    return render(request, template, context)
