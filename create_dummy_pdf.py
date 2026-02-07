from reportlab.pdfgen import canvas

c = canvas.Canvas("dummy.pdf")
c.drawString(100, 750, "Hello World from NexusMind!")
c.save()
