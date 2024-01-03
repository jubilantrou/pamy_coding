    # fig, axs = plt.subplots(4, 2, figsize=(40, 20))
    # for dof in range(3):
    #     line = []
    #     ax = axs[dof, 0]
    #     ax.set_xlabel(r'Iterations')
    #     ax.set_ylabel(r'Error $\delta \theta$ in degree')
    #     line_temp, = ax.plot(range(nr), error_ff_mean[:, dof]*180/math.pi, label=r'dof-{}, mean ff'.format(dof))
    #     line.append(line_temp)
    #     line_temp, = ax.plot(range(nr), error_ff_max[:, dof]*180/math.pi, label=r'dof-{}, max ff'.format(dof))
    #     line.append(line_temp)
    #     ax.axvline(x=nr_train, color = 'b')
    #     ax.grid()
    #     ax.legend(handles=line, loc=legend_position)

    #     line = []
    #     ax = axs[dof, 1]
    #     ax.set_xlabel(r'Iterations')
    #     ax.set_ylabel(r'Error $\delta \theta$ in degree')
    #     line_temp, = ax.plot(range(nr), error_fb_mean[:, dof]*180/math.pi, label=r'dof-{}, mean fb'.format(dof))
    #     line.append(line_temp)
    #     line_temp, = ax.plot(range(nr), error_fb_max[:, dof]*180/math.pi, label=r'dof-{}, max fb'.format(dof))
    #     line.append(line_temp)
    #     ax.axvline(x=nr_train, color = 'b')
    #     ax.grid()
    #     ax.legend(handles=line, loc=legend_position)
    
    # line = []
    # ax = axs[3, 0]
    # ax.set_xlabel(r'Iterations')
    # ax.set_ylabel(r'Error $\Delta s$ in m')
    # line_temp, = ax.plot(range(nr), error_ff_mean[:, 3], label=r'mean ff')
    # line.append(line_temp)
    # line_temp, = ax.plot(range(nr), error_ff_max[:, 3], label=r'max ff')
    # line.append(line_temp)
    # ax.axvline(x=nr_train, color = 'b')
    # ax.grid()
    # ax.legend(handles=line, loc=legend_position)

    # line = []
    # ax = axs[3, 1]
    # ax.set_xlabel(r'Iterations')
    # ax.set_ylabel(r'Error $\Delta s$ in m')
    # line_temp, = ax.plot(range(nr), error_fb_mean[:, 3], label=r'mean fb')
    # line.append(line_temp)
    # line_temp, = ax.plot(range(nr), error_fb_max[:, 3], label=r'max fb')
    # line.append(line_temp)
    # ax.axvline(x=nr_train, color = 'b')
    # ax.grid()
    # ax.legend(handles=line, loc=legend_position)

    # plt.savefig(figure_error + '/' + 'verify' + '.pdf')
    # plt.close()