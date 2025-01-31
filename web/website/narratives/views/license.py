from django.shortcuts import render


def index(request):
    template = "narratives/license.html"
    context = {
        'title_page': 'Software License'
    }
    return render(request, template, context)
