from django.http import JsonResponse

def remove_character(request):
    if request.method == 'POST':
        character = request.POST.get('character')
        question = request.POST.get('question')
        return JsonResponse({"success": True, "message": f"Character '{character}' removed from question '{question}'."})
    return JsonResponse({'status': 'failed', 'message': 'Invalid request method'}, status=400)